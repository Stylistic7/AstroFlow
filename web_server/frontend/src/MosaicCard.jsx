import * as React from 'react';
import { useNavigate, useParams } from "react-router-dom";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';

import MosaicOutline from "./MosaicOutline.jsx";

// TODO: Don't show MosaicCards for "mosaics" with just one panel?
export default function MosaicCard({stacks, targets, mosaics, mosaic_id}) {
  const mosaic = mosaics[mosaic_id];
  const navigate = useNavigate();

  return (
    <Card sx={{ maxWidth: 345 }}>
      <CardActionArea
        onClick={event => navigate(`/mosaic/${mosaic_id}`)}
      >
        <div id="mosaicPanel" style={{margin: "2em"}}>
          <MosaicOutline
            stacks={stacks}
            targets={targets}
            mosaics={mosaics}
            mosaic_id={mosaic_id}
          />
        </div>
        <CardContent>
          <Typography gutterBottom variant="h5" component="div">
            Mosaic {mosaic.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Image scale:&nbsp;
            {stacks[targets[mosaic.target_ids[0]].stack_ids[0]].image_scale.toFixed(2)}"/px
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}