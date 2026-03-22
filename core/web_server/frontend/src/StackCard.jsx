import { useContext } from "react";
import { useNavigate } from "react-router-dom";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';
import AstroFlowContext from './AstroFlowContext.jsx';
import { AstroFlowImage } from './AstroFlowImage.jsx';

export default function StackCard({ stack_id }) {
  const { stacks } = useContext(AstroFlowContext);
  const stack = stacks[stack_id];
  const navigate = useNavigate();

  return (
    <Card>
      <CardActionArea
        onClick={event => navigate(`/stack/${stack_id}`)}
      >
        <CardContent>
          {
            <AstroFlowImage
              objectType="stack"
              objectId={stack_id}
              size="thumbnail"
              style={{ height: "100%" }}
            />
          }
          <Typography gutterBottom variant="h5" component="div">
            Stack {stack.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Image scale: {stack.image_scale.toFixed(2)}"/px
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}