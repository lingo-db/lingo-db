sudo apt install -y -V ca-certificates lsb-release wget
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb
sudo apt install -y -V ./apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb
sudo sed -i'' -e 's,https://apache.bintray.com/,https://apache.jfrog.io/artifactory/,g' /etc/apt/sources.list.d/apache-arrow.sources
sudo apt update
sudo apt install -y -V libarrow-dev # For C++