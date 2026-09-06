
import { Box, Typography, IconButton, Tooltip } from '@mui/material';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import QuarantineToggle from './QuarantineToggle.jsx';

const FlatThumbnail = ({ stack, run, flatStack, selected, quarantined, onSelect, onNavigate }) => {
  const thumbnailUrl = flatStack
    ? `/thumbnail/run/${stack.stable_hash}/${run.stable_hash}/${flatStack.stable_hash}`
    : `/thumbnail/run/${stack.stable_hash}/${run.stable_hash}`;

  const stable_hash = flatStack
    ? flatStack.stable_hash
    : 'none';

  return (
    <Box
      onClick={onSelect}
      sx={{
        border: '2px solid transparent',
        outline: selected ? '2px solid #2196f3' : 'none',
        borderRadius: 2,
        cursor: 'pointer',
        flexShrink: 0,
        opacity: quarantined ? 0.4 : 1,
        position: 'relative',
      }}
    >
      <Box
        component="img"
        src={thumbnailUrl}
        loading="lazy"
        alt={`Flat correction thumbnail for ${stable_hash}`}
        sx={{
          width: 750,
          height: 450,
          objectFit: 'cover',
          borderRadius: 1,
          backgroundColor: '#111',
          // The whole thumbnail selects this flat (visual tie-breaking).
          // Inspecting the flat in detail is the zoom icon below, so the
          // primary click is never hijacked by navigation.
          cursor: 'pointer',
        }}
      />
      {flatStack && (
        <Box
          onClick={(e) => e.stopPropagation()}
          sx={{ position: 'absolute', top: 8, right: 8, display: 'flex', gap: 0.5 }}
        >
          {onNavigate && (
            <Tooltip title="Open this flat stack">
              <IconButton
                size="small"
                sx={{ bgcolor: 'rgba(0,0,0,0.5)' }}
                onClick={(e) => { e.stopPropagation(); onNavigate(flatStack.stable_hash); }}
              >
                <ZoomInIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          <QuarantineToggle
            objectType="stack"
            objectId={flatStack.stable_hash}
          />
        </Box>
      )}
      <Typography align="center" sx={{ py: 1 }}>
        {stable_hash}
        {flatStack?.oldest_date && (
          // Capture date — the first thing that separates candidate flats
          // (a years-stale flat from another optical train renders a
          // dramatic "correction" that has nothing to do with this light;
          // 2026-09-04). Same toDateString() style as the Run header above.
          <Typography component="span" sx={{ color: 'text.secondary', ml: 1 }}>
            — {new Date(flatStack.oldest_date).toDateString()}
          </Typography>
        )}
      </Typography>
    </Box>
  );
};

export default FlatThumbnail;
