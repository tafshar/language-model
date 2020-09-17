#!/bin/bash

subword-nmt learn-bpe -s 1000 < ${1} > rules.${1}.txt | \
 
subword-nmt apply-bpe -c rules.${1}.txt < ${2} > applied.${2} | \
 
cat applied.${2} | sed "s/@@//g"


