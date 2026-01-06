# Reolink RLC-520A & Jetson Orin Nano Setup Guide

#### This guide captures the "Laptop Bridge" workflow for initializing a Reolink camera on an isolated PoE switch connected to a Jetson Orin Nano.

# 1. Network Architecture

    Jetson (Wi-Fi): Connection to the internet/main network.

    Jetson (Ethernet/eno1): Static Gateway 10.42.0.1 providing DHCP.

    PoE Switch: Unmanaged, connecting the Jetson and the Camera.

# 2. Configuration Workflow (The "Laptop Bridge")

## Phase A: The Jetson Handshake

#### Create the Shared Connection:

`sudo nmcli con add type ethernet con-name "InternetShare" ifname eno1 ipv4.method shared`
`sudo nmcli con up InternetShare`


#### Enable IP Forwarding:
This allows the Jetson to act as a router for the camera.

#### Enable immediately
`sudo sysctl -w net.ipv4.ip_forward=1`

#### Make it permanent across reboots
`echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf`


#### Fix Docker/Firewall Conflicts:
Docker often blocks DHCP requests on the bridge. Force the ports open:

`sudo iptables -I INPUT 1 -i eno1 -p udp --dport 67:68 --sport 67:68 -j ACCEPT`
`sudo iptables -t nat -F`


## Phase B: Initialization

Connect Laptop: Plug a Windows/Mac laptop into the PoE switch **only AFTER** rules have been passed and firewall conflicts corrected

Run Reolink Client: Use the official client to "Find" the camera. Set the admin password.

Enable Ports: Navigate to Network Settings > Advanced > Port Settings. Toggle RTSP and HTTP to ON.

# 3. Watermark & OSD Removal

To ensure facial recognition accuracy (avoiding high-contrast edges from text):

Login: Access the camera via the laptop or Jetson browser at its assigned IP (e.g., 10.42.0.159).

Display Settings:

Watermark: Off.

Date/Time: Hide.

Camera Name: Hide.

Note: Apply this specifically to the Sub-stream (Fluent) if that is your inference source.

# 4. Port Forwarding (RTSP to Main Network)

To make the internal camera stream accessible via the Jetson's Wi-Fi IP:

### 1. Forward RTSP (554)
sudo iptables -t nat -A PREROUTING -p tcp --dport 554 -j DNAT --to-destination 10.42.0.159:554
sudo iptables -I FORWARD 1 -p tcp -d 10.42.0.159 --dport 554 -j ACCEPT

### 2. Forward Web UI (Optional - mapped to 8080)
sudo iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to-destination 10.42.0.159:80

### 3. Ensure return traffic is Masqueraded
sudo iptables -t nat -A POSTROUTING -j MASQUERADE


# 5. Persistence

To ensure these rules survive a reboot (and Docker restarts):

sudo apt-get install iptables-persistent
sudo netfilter-persistent save


# 6. Accessing the Stream

Internal (On Jetson): rtsp://admin:<pw>@10.42.0.159/Preview_01_sub

External (Via Wi-Fi): rtsp://admin:<pw>@<JETSON_WIFI_IP>:554/Preview_01_sub