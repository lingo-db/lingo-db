## Get it running

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
