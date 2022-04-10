#!/bin/bash

# source: https://stackoverflow.com/questions/3231804/in-bash-how-to-add-are-you-sure-y-n-to-any-command-or-alias
read -r -p "Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo "Cleaning up..."
    rm -rf ./output/*
else
    echo "Aborting."
fi
