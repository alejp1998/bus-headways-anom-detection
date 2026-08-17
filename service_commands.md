# EMT buses polling service

## Configuration

In order to change the service parameters, the systemd unit file must be edited. This can be done with:

```bash
vim /etc/systemd/system/docker.emt-buses.service
```

After modifying this file, `systemctl daemon-reload` should be called in order to notify systemd about the changes. Then, we can restart the service with:

```bash
systemctl restart docker.emt-buses.service
```

## Usage

### Starting/restarting/stopping the service

```bash
systemctl [start|restart|stop] docker.emt-buses.service
```

### Viewing the logs

```bash
docker logs -f emt-buses
```
