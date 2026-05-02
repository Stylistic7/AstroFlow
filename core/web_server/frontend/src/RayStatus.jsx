import React, { useState, useContext, useEffect } from "react";
import {
  Box,
  Menu,
  MenuItem,
  Typography,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Switch,
  Divider,
  FormControl,
  Select,
} from "@mui/material";
import { ExpandMore, ExpandLess, Circle } from "@mui/icons-material";
import AstroFlowContext from './AstroFlowContext.jsx';

const formatSpeed = (mbps) => {
  if (!mbps) return "?";
  if (mbps >= 1000) return `${mbps / 1000} Gbps`;
  return `${mbps} Mbps`;
};

const RayStatus = () => {
  const { rayStatus, sseConnected } = useContext(AstroFlowContext);

  const [anchorEl, setAnchorEl] = useState(null);
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

  if (!sseConnected) {
    return (
      <Box sx={{ display: "flex", alignItems: "center", padding: "8px" }}>
        <Circle sx={{ fontSize: 10, color: "error.light", mr: 1 }} />
        <Typography variant="body2" color="error.light">
          Disconnected
        </Typography>
      </Box>
    );
  }

  const hasCluster = Object.keys(rayStatus).length > 0;

  return (
    <>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          cursor: "pointer",
          padding: "8px",
          borderRadius: "8px",
        }}
        onClick={(event) => setAnchorEl(event.currentTarget)}
      >
        {!hasCluster
          ? <Typography variant="body2" color="textSecondary">No Cluster</Typography>
          : <>
              <Circle sx={{ fontSize: 10, color: rayStatus.unhealthyNodes > 0 ? "error.light" : "success.light", mr: 1 }} />
              <Typography variant="body2" sx={{ mr: 1 }}>
                {rayStatus.healthyNodes} {rayStatus.healthyNodes === 1 ? 'Node' : 'Nodes'},
              </Typography>
              <Typography variant="body2" sx={{ mr: 1 }}>
                {rayStatus.totalCPUs} CPUs, {Math.round(rayStatus.totalRAM)} GB RAM
              </Typography>
            </>
        }
        <IconButton size="small">{anchorEl ? <ExpandLess /> : <ExpandMore />}</IconButton>
      </Box>

      <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={() => setAnchorEl(null)}>
        {hasCluster && (
          <List dense>
            {rayStatus.nodes.map((node, index) => (
              <ListItem key={index}>
                <Circle sx={{ fontSize: 10, color: node.health === "healthy" ? "green" : "red", mr: 1 }} />
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
      </Menu>
    </>
  );
};

export default RayStatus;
