## To reproduce results with docker
```
make reproduce
```
Note that docker comes with a slight runtime overhead (passing `--privileged` helps a bit). Measurements in the paper were conducted using native builds.

## For developers

1. Install libs:
```bash
sudo apt-get install libjemalloc-dev libboost-dev \
                     libboost-filesystem-dev \
                     libboost-system-dev \
                     libboost-regex-dev \
                     python-dev \
                     autoconf \
                     flex \
                     bison
pip3 install requests moz_sql_parser numpy pandas
```
2. Checkout Repo
3. Build dependencies (`make dependencies`)
4. Build&Test (`make run-test`)
