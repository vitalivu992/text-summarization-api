alias docker-build='docker build --tag text-summarization-api:latest .'
alias docker-run='docker run -p 5000:5000 -v $PWD/cnn_dailymail:/app/cnn_dailymail -v $PWD/pegasus-cnn_dailymail:/app/pegasus-cnn_dailymail text-summarization-api:latest'
alias docker-build-run='docker-build; docker-run'