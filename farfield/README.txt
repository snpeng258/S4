Far-field scripts (S4 only)

=== run2: S4 single RCWA + per-order receiving screens (fast approximate finite spot) ===

  At grazing incidence (|θ|≥60°), diffraction orders land at separate x (e.g. m=0
  @ ~227 mm, m=+1 @ ~56 mm). A unified screen shows interference fringes ("waves");
  default screen_mode=stitched places one receiver per order at its landing x.

  Scripts: run_s4_farfield2.sh -> farfield_spot_conv_workflow.sh -> s4_farfield_spot_convolve.py
  Output: *_farfield_spot_conv.png (stitched discrete spots)

=== run3: S4 angular-spectrum multi-angle + per-order screens (accurate finite spot) ===

  Same per-order receiving-screen layout as run2; each order field is coherent sum
  over incidence angles on its screen (not full-field interference on one plane).

  Scripts: run_s4_farfield3.sh -> farfield_spot_spectrum_workflow.sh
  Output: *_farfield_spot_spectrum.png

  screen_mode: auto|stitched|orders (auto → stitched when |θ|≥60°)
  ORDER_HALF_WIDTH_MM=8  half-width of each order receiver

=== run4: S4 Floquet + elliptical aperture + ASR far-field ===

  Per-order: S4 Floquet delta in k -> Gaussian beam divergence σ_θ=λ/(πw₀)
  convolved in k-space -> propagate (z=40 mm) -> ASR -> stitched lab CCD.

  Scripts: run_s4_farfield4.sh -> farfield_asr_workflow.sh -> s4_farfield_asr.py
          (asr_propagate.py: k-conv + ScalarDiffraction_ASR_AP port)

  Default: FARFIELD_METHOD=asr stitches per-order k-conv + ASR spots on lab (u,v) CCD.
  FARFIELD_METHOD=fraunhofer  analytic Fraunhofer spots (lab CCD stitch).
  FARFIELD_METHOD=asr-legacy   old spatial-patch ASR (debug).
  OBS_SPAN_MM=4  optional half-span override for m=0 (u,v) grid.

  Output: {wl}nm_{θ}deg_az{φ}deg_{POL}_ap{U}x{V}um_z{Z}mm_farfield_asr.png

=== run5: HHG harmonic overlay ===

  Scripts: run_s4_farfield5.sh -> farfield_hhg_workflow.sh -> s4_farfield_hhg_panel.py
  Output: hhg800nm_H*-H*_wl*-*nm_*_overlay.png (+ .spots.json)

=== Legacy / debug ===

  farfield_workflow.sh + s4_farfield_reconstruct.py
    Single S4 GetWaves plane-wave Floquet (--screen-mode orders|unified).

=== Methods ===

  | Run | Engine                  | S4 runs | FFT/ASR | Finite spot |
  |-----|-------------------------|---------|---------|-------------|
  | 2   | S4 conv                 | 1       | no      | yes (approx) |
  | 3   | S4 angle spectrum       | N_theta | no      | yes (accurate) |
  | 4   | S4 Floquet + ASR        | 1       | ASR     | yes (ellipse aperture) |
  | 5   | S4 Floquet + ASR (HHG) | N_harm  | ASR     | yes (overlay) |

Run:
  cd farfield && chmod +x run_s4_farfield*.sh *_workflow.sh
  ./run_s4_farfield2.sh    # S4 convolution finite spot
  ./run_s4_farfield3.sh    # S4 multi-angle finite spot
  ./run_s4_farfield4.sh    # S4 Floquet + elliptical aperture + ASR
  ./run_s4_farfield5.sh    # HHG overlay

Outputs go to ../runs/farfield/ (png, waves txt, json).
ASR diagnostic npz dumps go to ../data/farfield/.
Materials: ../shared/ (set automatically via S4_MATERIALS_DB).

Env run2/3: SPOT_W0_UM, N_THETA, ...
Env run4: WL_NM, ANGLE_DEG, AZIMUTH_DEG, Z_MM, APERTURE_U_UM, APERTURE_V_UM,
          APERTURE_N, OBS_N, ASR_N_U, ASR_N_V, CCD_SIZE_MM, AXIS_HALF_MM,
          OBS_THETA2_DEG, OBS_PHI2_DEG (optional override), ORDER_R_THRESH, INTENSITY_SCALE
