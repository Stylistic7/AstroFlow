const divmod = (y, x) => {
  let div = Math.trunc(y / x);
  let rem = y % x;

  return [div, rem];
}

const ra_to_hms = (ra) => {
  let [hours, hours_remainder] = divmod(ra * 3600, 15 * 3600);
  let [minutes, minutes_remainder] = divmod(hours_remainder, 15 * 60);
  let seconds = Math.round(minutes_remainder / 15);

  return `${String(hours).padStart(2, '0')}hr ${String(minutes).padStart(2, '0')}' ${String(seconds).padStart(2, '0')}"`;
}

const dec_to_dms = (dec) => {
  let negative = dec < 0 ? '-' : '';
  dec = Math.abs(dec);
  let [degrees, degrees_remainder] = divmod(dec * 3600, 3600);
  degrees_remainder = Math.abs(degrees_remainder);
  let [minutes, seconds] = divmod(degrees_remainder, 60);
  seconds = Math.round(seconds);

  return `${negative}${String(degrees).padStart(2, '0')}º ${String(minutes).padStart(2, '0')}' ${String(seconds).padStart(2, '0')}"`
}

export { ra_to_hms, dec_to_dms };
