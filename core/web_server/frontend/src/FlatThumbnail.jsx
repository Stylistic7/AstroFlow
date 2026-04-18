
import { Box, Typography } from '@mui/material';
import QuarantineToggle from './QuarantineToggle.jsx';

const FlatThumbnail = ({ stack, run, flatStack, selected, quarantined, onSelect, onNavigate }) => {
  const thumbnailUrl = flatStack
    ? `/thumbnail/run/${stack.stable_hash}/${run.stable_hash}/${flatStack.stable_hash}`
    : `/thumbnail/stack/${stack.stable_hash}`;

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
        onClick={(e) => {
          if (onNavigate && flatStack) {
            e.stopPropagation();
            onNavigate(flatStack.stable_hash);
          }
        }}
        sx={{
          width: 750,
          height: 450,
          objectFit: 'cover',
          borderRadius: 1,
          backgroundColor: '#111',
          cursor: onNavigate && flatStack ? 'zoom-in' : 'pointer',
        }}
      />
      {flatStack && (
        <Box
          onClick={(e) => e.stopPropagation()}
          sx={{ position: 'absolute', top: 8, right: 8 }}
        >
          <QuarantineToggle
            objectType="stack"
            objectId={flatStack.stable_hash}
          />
        </Box>
      )}
      <Typography align="center" sx={{ py: 1 }}>
        {stable_hash}
      </Typography>
    </Box>
  );
};

export default FlatThumbnail;
