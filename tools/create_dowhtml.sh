#!/bin/sh

tree /var/www/html/csv/ -H csv -L 1 --noreport --charset utf-8 > /var/www/html/download.html