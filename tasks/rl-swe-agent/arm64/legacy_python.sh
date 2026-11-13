#!/bin/bash
# Native fallback for upstream environments requiring Python 3.5 or 3.6.
set -euxo pipefail
minor="$1"
case "$minor" in
  3.5) version=3.5.10; checksum=0f0fa8685c1dc1f1dacb0b4e7779796b90aef99dc1fa4967a71b9da7b57d4a28;
       pip_version=20.3.4; setuptools_version=50.3.2; wheel_version=0.35.1 ;;
  3.6) version=3.6.15; checksum=6e28d7cdd6dd513dd190e49bca3972e20fcf455090ccf2ef3f1a227614135d91;
       pip_version=21.3.1; setuptools_version=59.6.0; wheel_version=0.37.1 ;;
  *) exit 2 ;;
esac
test "$(uname -m)" = aarch64
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl make gcc g++ perl \
  zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
  libncurses-dev uuid-dev
work=$(mktemp -d)
cd "$work"
# CPython 3.5/3.6 cannot use Ubuntu 22.04's OpenSSL 3 API.
curl --fail --location --retry 3 -o openssl.tar.gz \
  https://www.openssl.org/source/old/1.1.1/openssl-1.1.1w.tar.gz
echo 'cf3098950cb4d853ad95c0841f1f9c6d3dc102dccfcacd521d93925208b76ac8  openssl.tar.gz' | sha256sum -c -
tar -xf openssl.tar.gz
cd openssl-1.1.1w
./Configure linux-aarch64 shared --prefix=/opt/legacy-openssl --openssldir=/etc/ssl
make -j"${MAX_JOBS:-4}"
make install_sw
cd "$work"
curl --fail --location --retry 3 -o python.tar.xz \
  "https://www.python.org/ftp/python/$version/Python-$version.tar.xz"
echo "$checksum  python.tar.xz" | sha256sum -c -
tar -xf python.tar.xz
cd "Python-$version"
CPPFLAGS=-I/opt/legacy-openssl/include \
LDFLAGS='-L/opt/legacy-openssl/lib -Wl,-rpath,/opt/legacy-openssl/lib' \
CFLAGS='-O2 -fcommon -fwrapv' ./configure --prefix="/opt/legacy-python/$minor" --without-ensurepip
make -j"${MAX_JOBS:-4}"
make install
ln -sf python3 "/opt/legacy-python/$minor/bin/python"
python="/opt/legacy-python/$minor/bin/python3"
"$python" -c 'import ssl, ctypes, sqlite3, bz2, lzma; print(ssl.OPENSSL_VERSION)'
"$python" -m ensurepip
"$python" -m pip install "pip==$pip_version" "setuptools==$setuptools_version" "wheel==$wheel_version"
cd /
rm -rf "$work" /var/lib/apt/lists/*
