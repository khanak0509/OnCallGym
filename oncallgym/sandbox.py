"""Run agent shell commands as a separate, unprivileged Unix user.

The agent may read the deployment (configs, logs, databases) and edit config files in
etc/, but it must not be able to write databases, the email outbox, logs, code, or
verifier material, and it must not be able to signal service processes. That is
enforced by the OS: services run as the stack owner, agent commands run as AGENT_USER.

Permissions applied to an episode home (owner = the user running the stack):
  home, data/, logs/, run/          0755 (agent can read, not write)
  data/* files                      0644
  etc/ and subdirs                  0777 + files 0666 (agent may edit and add files)
  private dir                       0700 (agent cannot read)

How we switch user: as root (Kaggle) we setuid in the child; otherwise via
`sudo -n -u agent` (the dev container grants exactly that).
"""

import os
import pwd
import shutil
import subprocess

AGENT_USER = os.environ.get("ONCALLGYM_AGENT_USER", "agent")
MAX_OUTPUT = 2000
DEFAULT_TIMEOUT_S = 10.0


class SandboxUnavailable(RuntimeError):
    pass


def agent_ids():
    try:
        pw = pwd.getpwnam(AGENT_USER)
    except KeyError:
        return None
    return pw.pw_uid, pw.pw_gid


def ensure_agent_user():
    """Create the agent user when running as root (Kaggle VMs). In the dev container the
    image already has it."""
    if agent_ids() is not None:
        return
    if os.geteuid() != 0:
        raise SandboxUnavailable(f"user {AGENT_USER!r} missing and not root")
    subprocess.run(
        ["useradd", "--no-create-home", "--shell", "/bin/sh", AGENT_USER],
        check=True,
        capture_output=True,
    )


def mode() -> str:
    if agent_ids() is None:
        raise SandboxUnavailable(f"user {AGENT_USER!r} does not exist")
    if os.geteuid() == 0:
        return "setuid"
    if shutil.which("sudo"):
        r = subprocess.run(["sudo", "-n", "-u", AGENT_USER, "true"], capture_output=True)
        if r.returncode == 0:
            return "sudo"
    raise SandboxUnavailable("cannot switch to the agent user (no root, no sudo rule)")


def lock_down(home: str):
    for root, _dirs, files in os.walk(home):
        rel = os.path.relpath(root, home)
        top = rel.split(os.sep)[0]
        if top == "etc":
            os.chmod(root, 0o777)  # no sticky bit: sed -i style edits must work
            for f in files:
                os.chmod(os.path.join(root, f), 0o666)
        else:
            os.chmod(root, 0o755)
            for f in files:
                p = os.path.join(root, f)
                if not os.path.islink(p):
                    os.chmod(p, 0o644)
    # the private TLS key must stay readable by auth but it is a test-only key; what the
    # agent must not have is verifier material, which lives outside home (0700)


def run(command: str, cwd: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Run one shell command as the agent user. Output is merged and truncated."""
    m = mode()
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "TERM": "dumb",
    }
    # `timeout` runs as the agent, so it can kill everything the command started
    inner = ["timeout", "-s", "KILL", str(int(timeout_s)), "/bin/sh", "-c", command]
    if m == "sudo":
        argv = ["sudo", "-n", "-u", AGENT_USER, "--", *inner]
        preexec = None
    else:
        uid, gid = agent_ids()
        argv = inner

        def preexec():
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)

    try:
        p = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=timeout_s + 5,
            preexec_fn=preexec,
            start_new_session=True,
        )
        out = (p.stdout + p.stderr).decode(errors="replace")
        code = p.returncode
        timed_out = code in (124, 137, -9)
        if timed_out:
            code = None
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"") + (e.stderr or b"")).decode(errors="replace")
        code = None
        timed_out = True
    truncated = len(out) > MAX_OUTPUT
    if truncated:
        out = out[: MAX_OUTPUT // 2] + "\n...[truncated]...\n" + out[-MAX_OUTPUT // 2 :]
    return {"exit_code": code, "timed_out": timed_out, "truncated": truncated, "output": out}
