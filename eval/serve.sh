#!/bin/bash
PORT=${1:-8081}
echo "Starting eval UI at http://localhost:$PORT"
cd "$(dirname "$0")"
php -S 0.0.0.0:$PORT ui.php
