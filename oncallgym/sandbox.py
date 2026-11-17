"""Run agent shell commands as a separate, unprivileged Unix user.

The agent may read the deployment (configs, logs, databases) and edit config files in
etc/, but it must not be able to write databases, the email outbox, logs, code, or
verifier material, and it must not be able to signal service processes. That is
enforced by the OS: services run as the stack owner, agent commands run as an agent user.

Each concurrent episode (port block) has its own agent user (agent0..agent15), so after
every command we can SIGKILL all of that user's processes: nothing the agent starts
(background jobs, setsid'd daemons) outlives the command. At episode start and end the
user's files in /tmp are wiped, so nothing carries over between episodes.

Permissions applied to an episode home (owner = the user running the stack):
  home, data/, logs/, run/          0755 (agent can read, not write)
  data/* and logs/* files           0644
  run/internal.token                0600 (service-to-service token, agent cannot read)
  etc/ and subdirs                  0777 + files 0666 (agent may edit and add files)
  private dir                       0700 (agent cannot read)

How we switch user: as root (Kaggle) we setuid in the child; otherwise via
`sudo -n -u agentN` (the dev container grants exactly that).
"""

import os
import pwd
import shutil
import subprocess

AGENT_PREFIX = os.environ.get("ONCALLGYM_AGENT_PREFIX", "agent")
MAX_AGENT_USERS = 16  # concurrent episodes per machine
MAX_OUTPUT = 2000
DEFAULT_TIMEOUT_S = 10.0
PRIVATE_FILES = {os.path.join("run", "internal.token")}


class SandboxUnavailable(RuntimeError):
    pass


def agent_user(block: int) -> str:
    return f"{AGENT_PREFIX}{block % MAX_AGENT_USERS}"


def agent_ids(user: str):
    try:
        pw = pwd.getpwnam(user)
    except KeyError:
        return None
    return pw.pw_uid, pw.pw_gid


def ensure_agent_user(user: str):
    """Create the agent user when running as root (Kaggle VMs). The dev container image
    already has agent0..agent15."""
    if agent_ids(user) is not None:
        return
    if os.geteuid() != 0:
        raise SandboxUnavailable(f"user {user!r} missing and not root")
    subprocess.run(
        ["useradd", "--no-create-home", "--shell", "/bin/sh", user],
        check=True,
        capture_output=True,
    )


_MODE: dict[str, str] = {}


def mode(user: str) -> str:
    if user in _MODE:
        return _MODE[user]
    if agent_ids(user) is None:
        raise SandboxUnavailable(f"user {user!r} does not exist")
    if os.geteuid() == 0:
        m = "setuid"
    elif (
        shutil.which("sudo")
        and subprocess.run(["sudo", "-n", "-u", user, "true"], capture_output=True).returncode == 0
    ):
        m = "sudo"
    else:
        raise SandboxUnavailable("cannot switch to the agent user (no root, no sudo rule)")
    _MODE[user] = m
    return m


def home_of(user: str) -> str:
    return f"/tmp/oncallgym-agent-home/{user}"


def _as_user(user: str, argv: list[str], cwd: str, env: dict, timeout: float):
    if mode(user) == "sudo":
        return subprocess.run(
            ["sudo", "-n", "-u", user, "--", *argv],
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=timeout,
            start_new_session=True,
        )
    uid, gid = agent_ids(user)

    def preexec():
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        timeout=timeout,
        preexec_fn=preexec,
        start_new_session=True,
    )


def kill_all(user: str):
    """SIGKILL every process of the agent user."""
    for _ in range(2):
        _as_user(user, ["/bin/sh", "-c", "kill -9 -1 2>/dev/null; true"], "/", {}, 10)


def wipe(user: str):
    """Kill the agent user's processes and delete everything it left in temp dirs."""
    kill_all(user)
    _as_user(
        user,
        [
            "/bin/sh",
            "-c",
            'find /tmp /var/tmp /dev/shm -xdev -user "$(id -un)" -depth -delete 2>/dev/null; true',
        ],
        "/",
        {"PATH": "/usr/bin:/bin"},
        30,
    )


def protect_code(user: str):
    """Make the code root (northwind/, oncallgym/, training/) unreadable to agent users:
    only the stack owner, which runs the services and the verifier, may read it. Without
    this an agent could import the fault definitions or regenerate the pristine configs.
    Verified by trying as the agent user (some bind mounts ignore modes); raises if the
    code is still readable."""
    import northwind

    root = os.path.dirname(os.path.dirname(os.path.abspath(northwind.__file__)))
    if root in _PROTECTED:
        return root
    try:
        os.chmod(root, 0o700)
    except PermissionError:
        pass
    probe = _as_user(
        user,
        ["/bin/sh", "-c", f"ls {root}/oncallgym >/dev/null 2>&1 && echo READABLE"],
        "/",
        {"PATH": "/usr/bin:/bin"},
        10,
    )
    if b"READABLE" in probe.stdout:
        raise SandboxUnavailable(
            f"code at {root} is readable by agent users; keep the repo under an owner-only "
            "directory (the dev container mounts it at /home/dev/OnCallGym)"
        )
    _PROTECTED.add(root)
    return root


_PROTECTED: set[str] = set()


def lock_down(home: str, user: str):
    protect_code(user)
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
                if os.path.islink(p):
                    continue
                private = os.path.relpath(p, home) in PRIVATE_FILES
                os.chmod(p, 0o600 if private else 0o644)


def run(user: str, command: str, cwd: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Run one shell command as the agent user. Output is merged and truncated. Every
    process the command started is killed afterwards."""
    home = home_of(user)
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": home,
        "LANG": "C.UTF-8",
        "TERM": "dumb",
    }
    script = f"mkdir -p {home} 2>/dev/null; {command}"
    inner = ["timeout", "-s", "KILL", str(int(timeout_s)), "/bin/sh", "-c", script]
    try:
        p = _as_user(user, inner, cwd, env, timeout_s + 5)
        out = (p.stdout + p.stderr).decode(errors="replace")
        code = p.returncode
        timed_out = code in (124, 137, -9)
        if timed_out:
            code = None
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"") + (e.stderr or b"")).decode(errors="replace")
        code = None
        timed_out = True
    finally:
        kill_all(user)
    truncated = len(out) > MAX_OUTPUT
    if truncated:
        out = out[: MAX_OUTPUT // 2] + "\n...[truncated]...\n" + out[-MAX_OUTPUT // 2 :]
    return {"exit_code": code, "timed_out": timed_out, "truncated": truncated, "output": out}
