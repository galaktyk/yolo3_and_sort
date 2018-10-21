echo "Video stream checking will run in 150 sec"
sleep 150
while [ 1 ];do
  
  size_old=$(ls -l /home/batman/proj/deep_sort_yolov3/gst.jpg | cut -d " " -f5)
  sleep 6

  size_now=$(ls -l /home/batman/proj/deep_sort_yolov3/gst.jpg | cut -d " " -f5)
  
  if [ $size_old -eq $size_now ];then
    time=$(date)
    text="Stream not response killing gst-launch-1.0 and restart"  
    echo "${bold}$time | $text ${normal}"
    pkill gst-launch-1.0
    echo "$time | $text" >> log.md
    sleep 20






  else 


  time=$(date)

  echo "${bold}$time | Stream fine ${normal}"
  fi
done
    
