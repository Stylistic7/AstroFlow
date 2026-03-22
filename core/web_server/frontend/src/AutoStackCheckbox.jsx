import { useContext } from "react";
import { Checkbox, FormGroup, FormControlLabel } from '@mui/material';

import AstroFlowContext from './AstroFlowContext.jsx';

export default function AutoStackCheckbox({ object_type, object_id }) {
  const { shoppingCart } = useContext(AstroFlowContext);

  const autoStackButtonChanged = (event) => {
    if (event.target.checked) {
      fetch(`/dispatch`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          type: 'add_to_shopping_cart', item: [object_type, object_id]
        })
      })
    } else {
      fetch(`/dispatch`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          type: 'remove_from_shopping_cart', item: [object_type, object_id]
        })
      })
    }
  };

  const in_shopping_cart = () => {
    for (const [t, i] of shoppingCart) {
      if (t === object_type && i === object_id) {
        return true;
      }
    }

    return false;
  };

  return (
    <FormGroup>
      <FormControlLabel control={
        <Checkbox
          checked={in_shopping_cart()}
          onChange={autoStackButtonChanged}
          size="small" />
      } label="Auto-Stack" />
    </FormGroup>
  );
}