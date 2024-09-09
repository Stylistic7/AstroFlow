import React, { useContext } from "react";
import { IconButton, Tooltip } from '@mui/material';
import DirectionsRunIcon from '@mui/icons-material/DirectionsRun';

import AstroFlowContext from './AstroFlowContext.jsx';
import ToggleIconButton from './ToggleIconButton.jsx';


export default function ProcessingActionButtons({ object_type, object_id }) {
  const { shoppingCart } = useContext(AstroFlowContext);
  console.log('shoppingCart is', shoppingCart);

  const exists = shoppingCart.some(
    item => item[0] === object_type && item[1] === object_id);

  const handleChange = (name, checked) => {
    if (checked) {
      fetch(`/add_to_shopping_cart/${object_type}/${object_id}`);
    } else {
      fetch(`/remove_from_shopping_cart/${object_type}/${object_id}`);
    }
  };

  return <>
    <Tooltip title="Process Automatically">
      <IconButton color="white">
        <ToggleIconButton
          name="play"
          icon={<DirectionsRunIcon />}
          defaultValue={exists}
          handleChange={handleChange}
        />
      </IconButton>
    </Tooltip>
  </>;
}