import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';
import { useNavigate, useParams } from "react-router-dom";

import MosaicOutline from "./MosaicOutline.jsx";

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
            Lizards are a widespread group of squamate reptiles, with over 6,000
            species, ranging across all continents except Antarctica
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}