// The canonical Pipeline / Tap / Filter pill rail. Data source is
// /api/pipeline_outputs/<object_type>/<object_id>. See
// [[project_url_grammar]].
import React, { useContext, useEffect, useState } from 'react';
import { Box, Chip, Menu, MenuItem } from '@mui/material';
import AstroFlowContext from './AstroFlowContext.jsx';
import { defaultDisplayFor } from './urls.js';


function unique(xs) {
  return Array.from(new Set(xs));
}


function Pill({ label, options, value, onChange }) {
  const [anchor, setAnchor] = useState(null);
  return (
    <>
      <Chip
        label={value ? `${label}: ${value}` : label}
        size="small"
        onClick={(e) => setAnchor(e.currentTarget)}
        disabled={options.length === 0}
        sx={{ mr: 1 }}
      />
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}>
        {options.map((opt) => (
          <MenuItem
            key={opt}
            selected={opt === value}
            onClick={() => { setAnchor(null); onChange(opt); }}
          >
            {opt}
          </MenuItem>
        ))}
      </Menu>
    </>
  );
}


export default function ViewPills({ hierarchy, value, onChange }) {
  // The deepest hierarchy entry drives the pipeline_outputs lookup.
  const tail = hierarchy[hierarchy.length - 1];
  const { computeGraphStatus } = useContext(AstroFlowContext);
  const [data, setData] = useState({ filters: [], pipelines: [] });

  useEffect(() => {
    if (!tail) return;
    let cancelled = false;
    fetch(`/api/pipeline_outputs/${tail.type}/${tail.id}`)
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (!cancelled && d) setData(d); })
      .catch((e) => console.warn(e));
    return () => { cancelled = true; };
  }, [tail?.type, tail?.id, computeGraphStatus]);

  // Flatten the (pipeline, taps, outputs) tree into one (p, d, f) list.
  const availableViews = [];
  for (const p of data.pipelines || []) {
    for (const tap of p.taps || []) {
      for (const out of tap.outputs || []) {
        if (out.exists) availableViews.push({
          pipeline: p.name, display: tap.display, filter: out.filter,
        });
      }
    }
  }

  const pipelines = unique(availableViews.map(v => v.pipeline));
  const displaysForPipeline = value?.pipeline
    ? unique(availableViews.filter(v => v.pipeline === value.pipeline).map(v => v.display))
    : [];
  const filtersForTap = value?.pipeline && value?.display
    ? unique(availableViews.filter(v => v.pipeline === value.pipeline && v.display === value.display).map(v => v.filter))
    : [];

  const pickPipeline = (p) => {
    const displays = unique(availableViews.filter(v => v.pipeline === p).map(v => v.display));
    const display = displays.includes(value?.display) ? value.display
                  : (displays.includes(defaultDisplayFor(tail.type)) ? defaultDisplayFor(tail.type)
                  : displays[0]);
    const filters = unique(availableViews.filter(v => v.pipeline === p && v.display === display).map(v => v.filter));
    const filter = filters.includes(value?.filter) ? value.filter : filters[0];
    onChange({ pipeline: p, display, filter });
  };
  const pickDisplay = (d) => {
    const filters = unique(availableViews.filter(v => v.pipeline === value.pipeline && v.display === d).map(v => v.filter));
    const filter = filters.includes(value?.filter) ? value.filter : filters[0];
    onChange({ ...value, display: d, filter });
  };
  const pickFilter = (f) => onChange({ ...value, filter: f });

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', p: 1 }}>
      <Pill label="Pipeline" options={pipelines} value={value?.pipeline} onChange={pickPipeline} />
      <Pill label="Tap" options={displaysForPipeline} value={value?.display} onChange={pickDisplay} />
      <Pill label="Filter" options={filtersForTap} value={value?.filter} onChange={pickFilter} />
    </Box>
  );
}
