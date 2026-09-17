import React, { useContext, useMemo } from 'react';
import { Box, IconButton, Paper } from '@mui/material';
import { OpenInNew } from '@mui/icons-material';
import MetadataItem from './MetadataItem.jsx';
import DecisionBadge from './DecisionBadge.jsx';
import ConcernsBadge from './ConcernsBadge.jsx';
import AstroFlowContext from './AstroFlowContext.jsx';

import { ra_to_hms, dec_to_dms, format_duration } from './units.js'

const StackHeader = ({
  stack_type, stack, stack_id, pipeline, finish,
}) => {
  const { isQuarantined, catalogEntries } = useContext(AstroFlowContext);

  // Catalog designations behind this object's canonical name, e.g.
  // "Lion Nebula" -> "Sh2-132", "Andromeda Galaxy" -> "NGC224 · M31".
  // Self-named entries (name === designation) carry no extra information.
  const catalogDesignations = useMemo(() => {
    const designations = (catalogEntries || [])
      .filter(([designation, name]) => name === stack.name && designation !== name)
      .map(([designation]) => designation);
    return designations.slice(0, 3).join(' · ');
  }, [catalogEntries, stack.name]);
  const handleOpenStack = async () => {
    // Opens the viewed finish's input taps — the CURRENT pipeline's master
    // (e.g. WBPP `stack_files` for "Hybrid: PI Master + OSS Mosaic"), not
    // some default pipeline's. No view yet ⇒ nothing to open.
    if (!pipeline || !finish) return;
    await fetch(
      `/api/open/${stack_type}/${stack_id}/${encodeURIComponent(pipeline)}/${encodeURIComponent(finish)}`,
      { method: 'POST' },
    );
  };

  // NO pipeline chips here. The breadcrumb bar already carries the pipeline
  // as one PillCrumb dropdown (AstroFlowBreadcrumbs), which is the canonical
  // URL grammar — hierarchy / pipeline / finish, one control per segment.
  // This header used to render a Chip PER PIPELINE beneath it, so a stack
  // with two producing pipelines showed two chips offering exactly the
  // choice the dropdown above already offered.
  return (
    <Paper
      elevation={3}
      sx={{
        pl: 2, pt: 0.5, pb: 0.5,
        opacity: isQuarantined(stack_type, stack_id) ? 0.5 : 1,
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 4, rowGap: 1 }}>
            {catalogDesignations && <MetadataItem label="Catalog" value={catalogDesignations} />}
            {(stack_type === 'stack') && <MetadataItem label="RA" value={ra_to_hms(stack.center[0])} />}
            {(stack_type === 'stack') && <MetadataItem label="Dec" value={dec_to_dms(stack.center[1])} />}
            {(stack_type === 'stack') && <MetadataItem label="Scale" value={`${stack.image_scale.toFixed(3)}"/px`} />}
            {(stack_type === 'stack') && <MetadataItem label="Angle" value={`${(stack.rotation_angle * (180.0 / Math.PI)).toFixed(1)}°`} />}
            <MetadataItem label="Camera" value={stack.camera} />
            {stack.camera_id_short && (
              <MetadataItem
                label="Camera ID"
                value={stack.camera_id_short}
                title={stack.camera_id_hash}
              />
            )}
            {(stack_type === 'stack') && <MetadataItem label="Filter" value={stack.filter} />}
            {(stack_type === 'calibration_stack') && <MetadataItem label="Exposure" value={format_duration(stack.exposure)} />}
            <MetadataItem label="Integration" value={format_duration(stack.integration)} />
            <MetadataItem label="Temp" value={`${stack.mean_temperature.toFixed(1)}°C`} />
          </Box>
        </Box>
        <Box sx={{ flexGrow: 1 }} />
        {stack_type === 'stack' && (
          <DecisionBadge objectType="stack" objectId={stack_id} />
        )}
        <ConcernsBadge objectType="stack" objectId={stack_id} />
        <IconButton onClick={handleOpenStack}>
          <OpenInNew />
        </IconButton>
      </Box>
    </Paper>
  );
};

export default StackHeader;
