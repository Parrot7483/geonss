# TODOS
- [x] Use elevation angle for weighted least squares
- [x] Add simple tropospheric correction (Niell mapping function)
- [ ] Multi constellation clock bias needs more unknown parameters to be solved (e.g. 5 unknown for double constellation / ISB or GGT)
- [x] Clean up pylint
- [ ] Use Dilution of Precision (DOP)
- [x] Add tests cases for satellite positions
- [ ] Compute satellite positions in parallel
- [x] Compute pseudo ranges in parallel
- [ ] Fix georinex library to be able to read navigation messages from IGS
- [ ] Refactor rinexmanger, create a class for it
- [ ] Add ionospheric correction for single frequency measurements
- [ ] Use doppler measurements for weighted least squares and velocity estimation
- [ ] Add a `LICENSE` file


Meeting
- [ ] Satellite phase center offset and variation
- [x] Wet Tropospheric correction
- [ ] Inter system bias 
- [ ] Check the residual of WLS for error measurement / Maybe only at the last iteration. Max iterations 5 or 6