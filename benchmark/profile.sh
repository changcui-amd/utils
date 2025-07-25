time=${1:-"2"}

bash -x start_profile.sh && sleep ${time} && bash -x stop_profile.sh
