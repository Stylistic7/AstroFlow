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
import Sparkline from './microviz/Sparkline.jsx';
import DualSparkline from './microviz/DualSparkline.jsx';

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

  // Latest sample from each utilization stream — used both to render
  // the trailing "X% Y%" numbers and to drive the icon's tone when
  // memory pressure or object-store pressure crosses a threshold.
  const cpuHistory = rayStatus.cpuHistory || [];
  const memHistory = rayStatus.memHistory || [];
  const objHistory = rayStatus.objectStoreHistory || [];
  const cpuLatest = cpuHistory.length ? cpuHistory[cpuHistory.length - 1] : null;
  const memLatest = memHistory.length ? memHistory[memHistory.length - 1] : null;
  const objLatest = objHistory.length ? objHistory[objHistory.length - 1] : null;

  // Memory pressure thresholds. Scheduled-memory > 85% means Ray's
  // running out of room to place new tasks; > 95% means tasks are
  // probably starting to block. Object store > 75% means imminent
  // spill-to-disk (slow); > 90% means cluster is struggling. CPU
  // saturation alone isn't a red flag — busy is good.
  const isMemError = (memLatest ?? 0) > 95 || (objLatest ?? 0) > 90;
  const isMemWarning = (memLatest ?? 0) > 85 || (objLatest ?? 0) > 75;

  // Pick the tone — order matters, severest first. SSE disconnect
  // ("we have no data at all") is the most urgent.
  const tone =
    !sseConnected ? "error"
    : isMemError ? "error"
    : hasCluster && rayStatus.unhealthyNodes > 0 ? "warning"
    : isMemWarning ? "warning"
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
    // amber/red tone when an unhealthy node count, memory pressure,
    // or SSE disconnect fires, and a second dot would be redundant.
    //
    // Node display is "X/Y" only when something died — surfacing a
    // missing worker as a digit (in addition to the icon's tone) so
    // the user knows "I'm supposed to have 4 nodes; 1 is gone"
    // without having to open Activity Monitor.
    const healthy = rayStatus.healthyNodes;
    const total = rayStatus.totalNodes ?? healthy;
    const nodeText = total > healthy
      ? `${healthy}/${total} nodes`
      : `${healthy} ${healthy === 1 ? 'node' : 'nodes'}`;

    // Single combined chart. Two overlaid lines on shared [0, 100]
    // axes — CPU in blue (info), memory color-shifts with pressure
    // (green → amber → red). No labels, no inline percentages; the
    // user learns the colors at a glance, and the expanded dropdown
    // carries the fully-labeled view for when they want to read
    // specific numbers.
    const memColor =
      isMemError ? "#e57373"        // error.light
      : isMemWarning ? "#ffcc80"    // warning.light
      : "#81c784";                   // success.light
    const cpuColor = "#64b5f6";     // info.light

    summaryNode = (
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography variant="body2">
          {`${nodeText} · ${rayStatus.totalCPUs} CPUs · ${Math.round(rayStatus.totalRAM)} GB`}
        </Typography>
        {cpuHistory.length > 0 && (
          <DualSparkline
            primary={cpuHistory}
            secondary={memHistory}
            primaryColor={cpuColor}
            secondaryColor={memColor}
            width={96}
            height={32}
          />
        )}
      </Box>
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
        {hasCluster && cpuHistory.length > 0 && (
          <Box sx={{ px: 2, py: 1.5 }}>
            <DualSparkline
              primary={cpuHistory}
              secondary={memHistory}
              primaryColor="#64b5f6"
              secondaryColor={
                isMemError ? "#e57373"
                : isMemWarning ? "#ffcc80"
                : "#81c784"
              }
              width={280}
              height={80}
              strokeWidth={2}
            />
            <Box sx={{ display: "flex", justifyContent: "space-around", mt: 0.5 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <Box sx={{ width: 10, height: 2, backgroundColor: "#64b5f6" }} />
                <Typography variant="caption" color="text.secondary">
                  {`CPU ${Math.round(cpuLatest)}%`}
                </Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <Box sx={{
                  width: 10, height: 2,
                  backgroundColor: isMemError ? "#e57373"
                    : isMemWarning ? "#ffcc80"
                    : "#81c784",
                }} />
                <Typography variant="caption" color="text.secondary">
                  {`memory ${Math.round(memLatest)}%`}
                </Typography>
              </Box>
              {objLatest !== null && (
                <Typography variant="caption" color="text.secondary">
                  {`object store ${Math.round(objLatest)}%`}
                </Typography>
              )}
            </Box>
          </Box>
        )}
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
