
mod hartree_fock;

fn main() {

  // test calculation of a hubbard ring with 1000 sites and 1000 particles with Sz=0 and U/t=4

  let mut hf_calc = hartree_fock::HFState::init(1000,true,1,true,500,500,4.0,1e-9);
  hf_calc.mk_guess(0.5);
  hf_calc.new_den();
  hf_calc.do_scf(0.05,true);
}
