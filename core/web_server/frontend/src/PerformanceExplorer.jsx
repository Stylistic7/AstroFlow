import { useState, useEffect, useMemo } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Label,
} from 'recharts';

const MEGAPIXEL = 1_000_000;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatDuration(seconds) {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  return `${(seconds / 60).toFixed(1)} min`;
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <Paper sx={{ p: 1.5, fontSize: '0.85rem' }}>
      <div><strong>Pixels:</strong> {(d.megapixels).toFixed(1)} MP</div>
      <div><strong>Wall time:</strong> {formatDuration(d.wall_time_s)}</div>
      <div><strong>CPU time:</strong> {formatDuration(d.cpu_time_s)}</div>
      <div><strong>Peak RSS delta:</strong> {formatBytes(d.peak_rss_delta_b)}</div>
      <div><strong>Images:</strong> {d.image_count}</div>
      <div><strong>Node:</strong> {d.node_ip}</div>
    </Paper>
  );
}

function SummaryTable({ samples }) {
  if (!samples.length) return null;

  const wallTimes = samples.map(s => s.wall_time_s);
  const cpuTimes = samples.map(s => s.cpu_time_s);
  const rssDeltas = samples.map(s => s.peak_rss_delta_b);
  const megapixels = samples.map(s => s.megapixels);

  const stats = (arr) => ({
    min: Math.min(...arr),
    max: Math.max(...arr),
    median: [...arr].sort((a, b) => a - b)[Math.floor(arr.length / 2)],
    mean: arr.reduce((a, b) => a + b, 0) / arr.length,
  });

  const rows = [
    { label: 'Wall time', ...stats(wallTimes), fmt: formatDuration },
    { label: 'CPU time', ...stats(cpuTimes), fmt: formatDuration },
    { label: 'Peak RSS delta', ...stats(rssDeltas), fmt: formatBytes },
    { label: 'Input size', ...stats(megapixels), fmt: v => `${v.toFixed(1)} MP` },
  ];

  return (
    <TableContainer component={Paper} sx={{ mt: 2, mb: 3 }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Metric</TableCell>
            <TableCell align="right">Min</TableCell>
            <TableCell align="right">Median</TableCell>
            <TableCell align="right">Mean</TableCell>
            <TableCell align="right">Max</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map(row => (
            <TableRow key={row.label}>
              <TableCell>{row.label}</TableCell>
              <TableCell align="right">{row.fmt(row.min)}</TableCell>
              <TableCell align="right">{row.fmt(row.median)}</TableCell>
              <TableCell align="right">{row.fmt(row.mean)}</TableCell>
              <TableCell align="right">{row.fmt(row.max)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Typography variant="caption" sx={{ p: 1, display: 'block', color: 'text.secondary' }}>
        {samples.length} sample{samples.length !== 1 ? 's' : ''}
      </Typography>
    </TableContainer>
  );
}

export default function PerformanceExplorer() {
  const [data, setData] = useState(null);
  const [selectedFn, setSelectedFn] = useState('');
  const [yMetric, setYMetric] = useState('wall_time_s');

  useEffect(() => {
    fetch('/api/performance')
      .then(r => r.json())
      .then(d => {
        setData(d.functions);
        const names = Object.keys(d.functions).sort();
        if (names.length > 0 && !selectedFn) {
          setSelectedFn(names[0]);
        }
      });
  }, []);

  const functionNames = useMemo(
    () => (data ? Object.keys(data).sort() : []),
    [data]
  );

  const samples = useMemo(() => {
    if (!data || !selectedFn || !data[selectedFn]) return [];
    return data[selectedFn].map(s => ({
      ...s,
      megapixels: (s.image_totals?.pixels || 0) / MEGAPIXEL,
      image_count: s.image_totals?.count || 0,
      peak_rss_mb: s.peak_rss_delta_b / (1024 * 1024),
    }));
  }, [data, selectedFn]);

  const yAxisConfig = {
    wall_time_s: { label: 'Wall time (s)', key: 'wall_time_s' },
    cpu_time_s: { label: 'CPU time (s)', key: 'cpu_time_s' },
    peak_rss_mb: { label: 'Peak RSS delta (MB)', key: 'peak_rss_mb' },
  };

  const yConf = yAxisConfig[yMetric];

  if (!data) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography>Loading performance data...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1000 }}>
      <Typography variant="h5" gutterBottom>
        Performance Explorer
      </Typography>

      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 3 }}>
        <FormControl sx={{ minWidth: 300 }}>
          <InputLabel>Function</InputLabel>
          <Select
            value={selectedFn}
            label="Function"
            onChange={e => setSelectedFn(e.target.value)}
          >
            {functionNames.map(name => (
              <MenuItem key={name} value={name}>
                {name} ({data[name].length})
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <ToggleButtonGroup
          value={yMetric}
          exclusive
          onChange={(_, v) => v && setYMetric(v)}
          size="small"
        >
          <ToggleButton value="wall_time_s">Wall time</ToggleButton>
          <ToggleButton value="cpu_time_s">CPU time</ToggleButton>
          <ToggleButton value="peak_rss_mb">Memory</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {samples.length > 0 && (
        <>
          <SummaryTable samples={samples} />

          <Paper sx={{ p: 2 }}>
            <ResponsiveContainer width="100%" height={450}>
              <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis
                  type="number"
                  dataKey="megapixels"
                  name="Input size"
                  unit=" MP"
                  stroke="#999"
                >
                  <Label value="Input size (megapixels)" position="bottom" offset={10} fill="#999" />
                </XAxis>
                <YAxis
                  type="number"
                  dataKey={yConf.key}
                  name={yConf.label}
                  stroke="#999"
                >
                  <Label value={yConf.label} angle={-90} position="insideLeft" offset={-5} fill="#999" />
                </YAxis>
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Scatter
                  name={selectedFn}
                  data={samples}
                  fill="#90caf9"
                  opacity={0.7}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </Paper>
        </>
      )}

      {samples.length === 0 && selectedFn && (
        <Typography color="text.secondary">
          No performance samples recorded for {selectedFn}.
        </Typography>
      )}
    </Box>
  );
}
