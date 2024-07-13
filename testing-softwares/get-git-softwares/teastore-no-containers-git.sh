#!/usr/bin/env bash

# TEASTORE-NO-CONTAINERS:MAIN
rm -rf ../teastore-no-containers
git clone "https://github.com/ThaysonScript/teastore-no-containers.git"
mv teastore-no-containers ../teastore-no-containers

rm -rf ../teastore-no-containers/old_deprecated_codes
rm -rf ../teastore-no-containers/TeaStore_Dockerfiles
rm -rf ../teastore-no-containers/Tea/dockerfiles