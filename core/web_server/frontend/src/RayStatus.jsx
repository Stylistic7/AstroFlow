import React, { useState, useContext, useEffect } from "react";
import {
  Box,
  MenuItem,
  Typography,
  List,
  ListItem,
  ListItemText,
  Switch,
  Divider,
  FormControl,
  Select,
} from "@mui/material";
import Hub from "@mui/icons-material/Hub";
import AstroFlowContext from './AstroFlowContext.jsx';
import DropdownChip from './DropdownChip.jsx';
import StatusDot from './microviz/StatusDot.jsx';

const formatSpeed = (mbps) => {
  if (!mbps) return "?";
  if (mbps >= 1000) return `${mbps / 1000} Gbps`;
  return `${mbps} Mbps`;
};

const RayStatus = () => {
  const { rayStatus, sseConnected } = useContext(AstroFlowContext);

  const [isHeadNode, setIsHeadNode] = useState(false);
  const [clusterInterface, setClusterInterface] = useState("");
  const [interfaces, setInterfaces] = useState([]);

  useEffect(() => {
    fetch('/api/network_settings')
      .then(r => r.json())
      .then(j => {
        setIsHeadNode(j.ray_head_node_enabled);
        setClusterInterface(j.cluster_interface || "");
      })
      .catch(console.error);
    fetch('/api/network_interfaces')
      .then(r => r.json())
      .then(j => setInterfaces(j.interfaces || []))
      .catch(console.error);
  }, []);

  const patchSettings = (updates) => {
    fetch('/api/network_settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
  };

  const toggleHeadNode = () => {
    const newHeadState = !isHeadNode;
    setIsHeadNode(newHeadState);
    patchSettings({ ray_head_node_enabled: newHeadState });
  };

  const handleInterfaceChange = (e) => {
    const value = e.target.value;
    setClusterInterface(value);
    patchSettings({ cluster_interface: value || null });
  };

  const hasCluster = Object.keys(rayStatus).length > 0;

  // Pick the tone first — drives icon color AND the chip's background
  // tint. SSE disconnect is more urgent than an unhealthy node count
  // (the latter is data, the former means we have no data at all).
  const tone =
    !sseConnected ? "error"
    : hasCluster && rayStatus.unhealthyNodes > 0 ? "warning"
    : hasCluster ? "active"
    : "default";

  // Summary slot: a status dot + the most compressed cluster summary.
  // When disconnected we surface that explicitly so the chip self-
  // explains; when there's no cluster yet we say "no cluster" rather
  // than an empty string.
  let summaryNode;
  if (!sseConnected) {
    summaryNode = <Typography variant="body2" color="error.light">disconnected</Typography>;
  } else if (!hasCluster) {
    summaryNode = <Typography variant="body2" color="text.secondary">no cluster</Typography>;
  } else {
    // No leading StatusDot here — the Hub icon already carries the
    // amber/red tone when an unhealthy node count or SSE disconnect
    // fires, and a second dot saying the same thing is redundant.
    summaryNode = (
      <Typography variant="body2">
        {`${rayStatus.healthyNodes} ${rayStatus.healthyNodes === 1 ? 'node' : 'nodes'} · ${rayStatus.totalCPUs} CPUs · ${Math.round(rayStatus.totalRAM)} GB`}
      </Typography>
    );
  }

  return (
    <DropdownChip
      icon={<Hub />}
      summary={summaryNode}
      tone={tone}
      tooltipTitle="Ray cluster"
      ariaLabel="Ray cluster status"
    >
        {hasCluster && (
          <List dense>
            {rayStatus.nodes.map((node, index) => (
              <ListItem key={index}>
                <Box sx={{ mr: 1 }}>
                  <StatusDot tone={node.health === "healthy" ? "active" : "error"} />
                </Box>
                <ListItemText
                  primary={`${node.ip}: ${node.cpus} CPUs, ${Math.round(node.ram)} GB RAM`}
                  secondary={node.health === "unhealthy" ? node.message : ""}
                  sx={{ color: node.health === "unhealthy" ? "error.main" : "text.primary" }}
                />
              </ListItem>
            ))}
            <Divider />
          </List>
        )}
        <MenuItem>
          <Typography variant="body2" sx={{ flexGrow: 1 }}>
            Run Ray Head Node on this Machine
          </Typography>
          <Switch checked={isHeadNode} onChange={toggleHeadNode} />
        </MenuItem>
        <Divider />
        <MenuItem disableRipple sx={{ flexDirection: "column", alignItems: "stretch" }}>
          <Typography variant="body2" sx={{ mb: 0.5 }}>
            Cluster network interface
          </Typography>
          <FormControl size="small" fullWidth>
            <Select
              value={clusterInterface}
              onChange={handleInterfaceChange}
              displayEmpty
            >
              <MenuItem value="">
                <em>Auto (default route)</em>
              </MenuItem>
              {interfaces.map((iface) => (
                <MenuItem key={iface.name} value={iface.name}>
                  {iface.name} — {iface.ip} ({formatSpeed(iface.speed_mbps)})
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </MenuItem>
    </DropdownChip>
  );
};

export default RayStatus;
