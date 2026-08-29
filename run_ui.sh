#!/usr/bin/env bash
echo "========================================================"
echo " Starting VELTRAXX PS11 Lightweight Server"
echo "========================================================"
echo "Opening http://localhost:5000 ..."
python3 server.py || python server.py
