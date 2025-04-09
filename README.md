# TODOS

- [x] Use elevation angle for weighted least squares
- [x] Add simple tropospheric correction (Niell mapping function)
- [ ] Multi constellation clock bias needs more unknown parameters to be solved (e.g. 5 unknown for double constellation / ISB or GGT)
- [ ] Receiver clock bias should converge over time and not be calculated for each epoch independently
- [ ] Add ionospheric correction for single frequency measurements
- [x] Clean up pylint
- [ ] Refactor rinexmanger, create a class for it
- [ ] Use Kalman filter for position estimation
- [ ] Use Dilution of Precision (DOP)
- [ ] Use doppler measurements for weighted least squares and velocity estimation
- [ ] Add tests cases for satellite positions
- [ ] Add a `LICENSE` file
- [ ] Compute satellite positions in parallel
- [ ] Compute pseudo ranges in parallel
- [ ] Fix georinex library to be able to read navigation messages from IGS
