#!/usr/bin/env bash
set -euo pipefail
sudo mkdir -p /opt/powernap
sudo cp powernap.py powernap.conf requirements.txt /opt/powernap/
if [ ! -f /opt/powernap/powernap.conf ]; then
  sudo cp /opt/powernap/powernap.conf.example /opt/powernap/powernap.conf
fi
sudo chmod 755 /opt/powernap/powernap.py
sudo chmod 600 /opt/powernap/powernap.conf
sudo cp powernap.service /etc/systemd/system/powernap.service
sudo systemctl daemon-reload
echo "Installed. Run: sudo systemctl enable --now powernap.service"
