PORT=${1:-"8000"}

curl -s http://localhost:${PORT}/v1/completions -H "Content-Type: application/json" -d '{
  "prompt": "Hello, who are you",
  "temperature": 0,
  "top_k": 1
}'
