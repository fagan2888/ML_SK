#!/bin/bash

grep -B1 Mol $1 | grep -v Mol | grep -v \- | sort -n | tail -1
