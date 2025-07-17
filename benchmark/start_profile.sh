PORT=${1:-"8000"}

curl -s http://localhost:${PORT}/start_profile -H "Content-Type: application/json" -d '{}'
