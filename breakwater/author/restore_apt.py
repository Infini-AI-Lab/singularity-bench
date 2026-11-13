"""Build-only compatibility for legacy bases that removed their apt sources."""
from pathlib import Path
import platform

if not Path('/etc/apt/sources.list').exists() and not any(Path('/etc/apt/sources.list.d').glob('*')):
    release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    system = release['ID'].strip('"')
    suite = release['VERSION_CODENAME'].strip('"')
    if system == 'debian':
        lines = f'deb https://deb.debian.org/debian {suite} main\ndeb https://deb.debian.org/debian-security {suite}-security main\n'
    elif system == 'ubuntu':
        host = 'ports.ubuntu.com/ubuntu-ports' if platform.machine() == 'aarch64' else 'archive.ubuntu.com/ubuntu'
        lines = ''.join(f'deb https://{host} {suite}{suffix} main universe\n' for suffix in ('', '-updates', '-security'))
    else:
        raise SystemExit('Unsupported legacy base distribution')
    Path('/etc/apt/sources.list').write_text(lines)
