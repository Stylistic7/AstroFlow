import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';

export default function TargetCard({stacks, targets, mosaics, target_id}) {
  const target = targets[target_id];

  return (
    <Card sx={{ maxWidth: 345 }}>
      <CardActionArea>
        <CardMedia
          component="img"
          height="140"
          image="/static/images/cards/contemplative-reptile.jpg"
          alt="green iguana"
        />
        <CardContent>
          <Typography gutterBottom variant="h5" component="div">
            Target {target.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
              Stacks: {target.stack_ids.map(stack_id => 
                stacks[stack_id].filter
              )}
          </Typography>
          <Typography variant="body2" color="text.secondary">
              Image scale: {target.image_scale}
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}