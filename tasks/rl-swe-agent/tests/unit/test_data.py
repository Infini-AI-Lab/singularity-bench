import io
import tarfile
from pathlib import Path
import pytest
from prepare_data import extract_archive

def archive(extra=None):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w') as t:
        for name,data in [('instruction.md',b'do a task'),('task.toml',b'version="1.0"'),('environment/Dockerfile',b'FROM ubuntu:24.04'),('tests/test.sh',b'exit 0')]:
            m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o755 if name.endswith('.sh') else 0o644
            t.addfile(m,io.BytesIO(data))
        if extra:
            m,data=extra;t.addfile(m,io.BytesIO(data))
    return out.getvalue()

def test_valid_extract_modes_and_inventory(tmp_path):
    dest=tmp_path/'task';files=extract_archive(archive(),dest)
    assert len(files)==4
    assert (dest/'tests/test.sh').stat().st_mode & 0o111
    with pytest.raises(FileExistsError):extract_archive(archive(),dest)

@pytest.mark.parametrize('name',['../outside','/outside','a/../../outside','environment/../instruction.md','a\\b','instruction.md'])
def test_reject_bad_paths_or_collisions(tmp_path,name):
    m=tarfile.TarInfo(name);m.size=1
    with pytest.raises(ValueError):extract_archive(archive((m,b'x')),tmp_path/'task')
    assert not (tmp_path/'outside').exists()

def test_reject_symlink(tmp_path):
    m=tarfile.TarInfo('link');m.type=tarfile.SYMTYPE;m.linkname='/tmp'
    with pytest.raises(ValueError):extract_archive(archive((m,b'')),tmp_path/'task')

def test_extraction_size_bound(tmp_path):
    with pytest.raises(ValueError):extract_archive(archive(),tmp_path/'task',max_bytes=2)
