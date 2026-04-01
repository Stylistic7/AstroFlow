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
} from "@mui/material";
import { ExpandMore, ExpandLess, Circle } from "@mui/icons-material";
import AstroFlowContext from './AstroFlowContext.jsx';

const RayStatus = () => {
  const { rayStatus, sseConnected } = useContext(AstroFlowContext);

  const [anchorEl, setAnchorEl] = useState(null);
  const [isHeadNode, setIsHeadNode] = useState(false);

  useEffect(() => {
    fetch('/api/ray/head_node_enabled')
      .then(r => r.json())
      .then(j => setIsHeadNode(j.enabled))
      .catch(console.error);
  }, []);

  const toggleHeadNode = () => {
    const newHeadState = !isHeadNode;
    setIsHeadNode(newHeadState);
    fetch('/api/ray/head_node_enabled', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ enabled: newHeadState.toString() })
    })
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
      </Menu>
    </>
  );
};

export default RayStatus;
