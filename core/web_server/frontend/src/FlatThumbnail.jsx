
import { Box, Typography } from '@mui/material';

const FlatThumbnail = ({ stack, run, flatStack, selected, onSelect }) => {
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
        }}
      />
      <Typography align="center" sx={{ py: 1 }}>
        {stable_hash}
      </Typography>
    </Box>
  );
};

export default FlatThumbnail;
