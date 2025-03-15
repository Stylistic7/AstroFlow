import React, { useState } from "react";
import { Box, Typography, Accordion, AccordionSummary, AccordionDetails } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import CacheArtifactCard from "./CacheArtifactCard";

const merkleTreeData = {
  inputs: [
    {
      id: 1, target_id: "9b3ec47a98aa5ee8"
    },
    { id: 2, target_id: "85f42d0cb68b753f" },
  ],
  task: {
    name: "merge_images",
    config: { mode: "blend", size: "1024x1024" },
    log: "Task executed successfully. Blending images...",
  },
  output: {
    id: 3,
  },
};

const GraphVisualization = () => {
  const { inputs, task, output } = merkleTreeData;
  const [logExpanded, setLogExpanded] = useState(false);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, padding: 2 }}>
      {/* Input nodes */}
      <Box sx={{ display: "flex", gap: 2 }}>
        {inputs.map((input) => (
          <CacheArtifactCard target_id={input.target_id} />
        ))}
      </Box>

      {/* Task node */}
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 1,
          padding: 2,
          border: "1px solid gray",
          borderRadius: 1,
          width: "fit-content",
          backgroundColor: "gray",
        }}
      >
        <Typography variant="h6">Task: {task.name}</Typography>
        <Typography variant="body2">Mode: {task.config.mode}</Typography>
        <Typography variant="body2">Size: {task.config.size}</Typography>
        <Accordion expanded={logExpanded} onChange={() => setLogExpanded(!logExpanded)} sx={{ width: "100%" }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Logs</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2">{task.log}</Typography>
          </AccordionDetails>
        </Accordion>
      </Box>

      {/* Output node */}
      <CacheArtifactCard target_id="9d5fbb675c3156c0" />
    </Box>
  );
};

export default GraphVisualization;
