#!/bin/bash

fail=0
bold=$(tput bold)
normal=$(tput sgr0)
yoloed=0
while [ 1 ];do
while [ 1 ];do
  sudo ip -s -s neigh flush all
  time=$(date)
  text="Scanning for Raspberry Pi IP Address...(about 1-5 minute)"
  echo "${bold}$time | $text ${normal}"
  echo "$time | $text" >> /home/batman/proj/deep_sort_yolov3/log.md



  piip=$(sudo nmap -sn -T5 --min-parallelism 100 \
  --max-parallelism 256 10.105.128.0/18 | \
  awk '/^Nmap/{ipaddress=$NF}/B8:27:EB:38:D8/{print ipaddress}')
  if [ $piip ]
  then
    time=$(date)
    text="Found Raspberry Pi IP Address : ${piip}"
    echo "${bold}$time | $text ${normal}"
    echo "$time | $text" >> /home/batman/proj/deep_sort_yolov3/log.md
    
    

    break
  else
    
    time=$(date)
    text="Found no matching Raspberry Pi. sleep for 150 sec. and will try again."
    echo "${bold}$time | $text ${normal}"
    echo "$time | $text" >> /home/batman/proj/deep_sort_yolov3/log.md

    sleep 150
  fi
done

##___________________________________________________________________________


if [ $yoloed -eq 0 ];then
  xfce4-terminal -e "bash /home/batman/proj/deep_sort_yolov3/envscript.sh" -H
  xfce4-terminal -e "bash /home/batman/proj/deep_sort_yolov3/check_size.sh" -H
  (( yoloed++ ));
fi


##___________________________________________________________________________

while [ $fail -lt 3 ];do
sleep 3

time=$(date)
text="Start receiving video..."
echo "${bold}$time | $text ${normal}"
echo "$time | $text" >> log.md


gst-launch-1.0 -v tcpclientsrc host=$piip port=6006 \
 ! gdpdepay ! rtpjitterbuffer ! rtph264depay ! avdec_h264 ! jpegenc ! \
multifilesink post-messages=true \
location="/home/batman/proj/deep_sort_yolov3/gst.jpg"

echo "${bold}Can't receive any stream. sleep for 5 sec. and will try again."
sleep 5
(( fail+=1 ));

done

fail=0
echo "${bold}ERROR. Re-scanning for IP >>"

done

