import { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  List,
  ListItem,
  ListItemText,
  Typography,
  IconButton,
  Collapse,
  Box,
  Chip,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMore from '@mui/icons-material/ExpandMore';
import ExpandLess from '@mui/icons-material/ExpandLess';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

/**
 * Dialog listing concerns reported by the producer for an object.
 *
 * Each concern shows its title, severity, optional collapsible detail,
 * and any structured fields the emitter attached (frame counts,
 * filenames, ratios, etc.) as Chips.
 *
 * Props:
 *   open: boolean
 *   onClose: () => void
 *   concerns: array of concern dicts from /api/concerns
 */
export default function ConcernsDialog({ open, onClose, concerns }) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      onClick={(e) => e.stopPropagation()}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        Concerns
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        {concerns.length === 0 ? (
          <Typography color="text.secondary">No concerns to review.</Typography>
        ) : (
          <List disablePadding>
            {concerns.map((c, i) => (
              <ConcernItem key={i} concern={c} />
            ))}
          </List>
        )}
      </DialogContent>
    </Dialog>
  );
}

const RESERVED_FIELDS = new Set(["event", "detail", "_concern", "level", "timestamp"]);

function ConcernItem({ concern }) {
  const [expanded, setExpanded] = useState(false);
  const isError = concern._concern === "error";
  const Icon = isError ? ErrorOutlineIcon : WarningAmberIcon;
  const color = isError ? 'rgba(255,80,80,0.95)' : 'rgba(255,196,0,0.95)';

  const detail = concern.detail || "";
  const extras = Object.entries(concern).filter(([k]) => !RESERVED_FIELDS.has(k));
  const hasMore = detail || extras.length > 0;

  return (
    <ListItem
      sx={{
        flexDirection: 'column',
        alignItems: 'flex-start',
        borderBottom: '1px solid',
        borderColor: 'divider',
        py: 2,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', gap: 1 }}>
        <Icon sx={{ color, fontSize: 20 }} />
        <ListItemText
          primary={concern.event || "(untitled concern)"}
          secondary={concern.timestamp}
          sx={{ flex: 1, m: 0 }}
        />
        {hasMore && (
          <IconButton size="small" onClick={() => setExpanded(!expanded)}>
            {expanded ? <ExpandLess /> : <ExpandMore />}
          </IconButton>
        )}
      </Box>
      {hasMore && (
        <Collapse in={expanded} sx={{ width: '100%' }}>
          <Box sx={{ pl: 4, pt: 1 }}>
            {detail && (
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ whiteSpace: 'pre-wrap', mb: 1 }}
              >
                {detail}
              </Typography>
            )}
            {extras.length > 0 && (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {extras.map(([k, v]) => (
                  <Chip
                    key={k}
                    label={`${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
            )}
          </Box>
        </Collapse>
      )}
    </ListItem>
  );
}
