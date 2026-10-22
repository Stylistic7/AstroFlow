import { useContext } from "react";
import { useNavigate } from "react-router-dom";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';
import AstroFlowContext from './AstroFlowContext.jsx';
import { AstroFlowImage } from './AstroFlowImage.jsx';
import useShareReadiness from './useShareReadiness.js';

// The card's edge carries the share gate's verdict (metadata/share_gate.py):
// grey for a master that cannot be shared, amber for one shared with caveats,
// so a stack that needs attention is visible from the list, not only from
// its own page. The sharing button on the page says why.
const EDGE = { blocked: 'grey.600', caveats: 'warning.main' };

export default function StackCard({ stack_id }) {
  const { stacks } = useContext(AstroFlowContext);
  const stack = stacks[stack_id];
  const navigate = useNavigate();
  const readiness = useShareReadiness(stack_id);
  const verdict = readiness && (!readiness.ready ? 'blocked' : readiness.warnings.length ? 'caveats' : null);
  const note = verdict && (readiness.ready ? readiness.warnings : readiness.blocking).map((r) => r.title).join(' · ');

  return (
    <Card sx={verdict ? { borderLeft: 4, borderColor: EDGE[verdict] } : undefined}>
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
          {note && (
            <Typography variant="caption" sx={{ color: EDGE[verdict], display: 'block', mt: 0.5 }}>
              {verdict === 'blocked' ? 'Not ready to share: ' : 'Share with caveats: '}{note}
            </Typography>
          )}
        </CardContent>
      </CardActionArea>
    </Card>
  );
}
