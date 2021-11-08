
extern crate blas;
extern crate openblas_src;
use blas::dgemm;
use lapack::dsyev;

pub struct HFState {
// Things that are set as inputs on initalization
    nx:    usize, // nuber of sites in the x direction (grid is rectangular)
    ny:    usize, // number of sites in the y direction
    nocca: usize, // number of up spins
    noccb: usize, // number of down spins
    u:     f64,   // u/t which governs the on-site repulsion.
    tol:   f64,   // error tolerance for termination of the calculation

// Things that are nice to have around that are calulated from inputs
    nao:   usize, // number of particles
    lwork: usize, // size of the working array for lapack

    err: f64, // somewhere to store the calculated error for observation.

// Arrays that are used and the size will be set by the inputs
    dena:    Vec<f64>, // the single particle density matrix for up spins
    denb:    Vec<f64>, // the single particle density matrix for down spins
    focka:   Vec<f64>, // the fock matrix for up spins
    fockb:   Vec<f64>, // the fock matrix for down spins
    t:       Vec<f64>, // the nearest neighbor hopping matix
    eveca:   Vec<f64>, // the single up spin particle orbitals
    evecb:   Vec<f64>, // the single particle down spin orbitals
    work:    Vec<f64>, // thw working array for lapack
    errvec1: Vec<f64>, // arrays that the error are calculated with
    errvec2: Vec<f64>, // not really useful here but handy if I add DIIS later.
    evals:   Vec<f64>, // the orbital energies (kinda, really just temp values I have to put somewhere).
}


impl HFState {
    pub fn init(nx: usize, pbcx: bool, ny: usize, pbcy: bool, nocca: usize, noccb: usize, u: f64, tol: f64) -> HFState {
      HFState {
      nx:      usize::from(nx),
      ny:      usize::from(ny),
      nocca:   usize::from(nocca),
      noccb:   usize::from(noccb),
      u:       f64::from(u),
      tol:     f64::from(tol),
      nao:     usize::from(nx*ny),
      lwork:   usize::from(34*nx*ny),
      err:     1.0,
      dena:    vec![0.0; nx*ny*nx*ny as usize],
      denb:    vec![0.0; nx*ny*nx*ny as usize],
      focka:   vec![0.0; nx*ny*nx*ny as usize],
      fockb:   vec![0.0; nx*ny*nx*ny as usize],
      t:       mk_t(nx,ny,pbcx,pbcy),
      eveca:   vec![0.0; nx*ny*nx*ny as usize],
      evecb:   vec![0.0; nx*ny*nx*ny as usize],
      work:    vec![0.0; 34*nx*ny],
      errvec1: vec![0.0; nx*ny*nx*ny as usize],
      errvec2: vec![0.0; nx*ny*nx*ny as usize],
      evals:   vec![0.0; nx*ny as usize],
      }
    }


// updates the density matricies from the evecs
    pub fn new_den(&mut self) {
      unsafe {
      dgemm(b'N',b'T',self.nao as i32, self.nao as i32, self.nocca as i32, 1.0,
            &self.eveca, self.nao as i32, &self.eveca, self.nao as i32, 0.0,
      &mut self.dena, self.nao as i32);
      dgemm(b'N',b'T',self.nao as i32, self.nao as i32, self.noccb as i32, 1.0,
            &self.evecb, self.nao as i32, &self.evecb, self.nao as i32, 0.0,
            &mut self.denb, self.nao as i32);
      }
    }

// Calculate the mean-field energy of the current state
    pub fn get_energy(&self) -> f64 {
      let mut energy: f64 = 0.0;
      for i in 0..self.nao {
        energy += self.u*self.dena[s2i(i,i,self.nao)]*self.denb[s2i(i,i,self.nao)];
        for j in 0..self.nao {
          energy += self.t[s2i(i,j,self.nao)]*(self.dena[s2i(j,i,self.nao)]+self.denb[s2i(j,i,self.nao)]);
        }
      }
      energy
    }

// construct an inital guess. This will be a liniear combination of the two trivially solvable cases.
    pub fn mk_guess(&mut self, add_neel: f64) {
      let add_core = 1.0-add_neel;
      let mut ko: usize = 0;
      let mut kv: usize = 0;
      let mut info: i32 = 0;
      let mut i: usize = 0;
      let mut j: usize;

// This gets the "core" or U=0 solution which is just the eigenvectors of t.

      self.eveca = self.t.to_vec();

      unsafe {
        dsyev(b'V', b'U', self.nao as i32, &mut self.eveca, self.nao as i32,
              &mut self.evals, &mut self.work, self.lwork as i32, &mut info);
      }

// I'm sure there's some 'proper' way to do this with iterators,
// but I didn't spend the time to figure it out.
// I went with the Fortrany method since that is what I know.
      for i in 0..(self.nao*self.nao) {
        self.eveca[i] *= add_core;
      }

      self.evecb = self.eveca.to_vec();

// This jumble of unplesant indicies constructs the "neel" or the U=Inf solution
// where the spins alternate in a checkerboard pattern
      while i < self.nx {
        j = 0;
        while j < self.ny {
          if (i+j)%2 == 1 {
            self.eveca[s2i(s2i(i,j,self.nx),ko,self.nao)]            += add_neel;
            self.evecb[s2i(s2i(i,j,self.nx),ko+self.nao/2,self.nao)] += add_neel;
            ko += 1;
          } else {
            self.eveca[s2i(s2i(i,j,self.nx),kv+self.nao/2,self.nao)] += add_neel;
            self.evecb[s2i(s2i(i,j,self.nx),kv,self.nao)]            += add_neel;
            kv += 1;
          }
          j += 1;
        }
        i += 1;
      }

      self.new_den();
    }

// This construcs the fock matrix from the density, u, and t. (and see my earlier comment about iterators)
    pub fn mk_fock(&mut self, damping: f64, do_uhf: bool) {

// The damping factor can help stability in unstable scenarios.
      for i in 0..(self.nao*self.nao) {
        self.focka[i] = damping*self.focka[i] + self.t[i];
        self.fockb[i] = damping*self.fockb[i] + self.t[i];
      }

      for i in 0..self.nao {
        self.focka[s2i(i,i,self.nao)] += self.u*self.denb[s2i(i,i,self.nao)];
        self.fockb[s2i(i,i,self.nao)] += self.u*self.dena[s2i(i,i,self.nao)];
      }

      if !do_uhf {
        for i in 0..(self.nao*self.nao) {
          self.focka[i] += self.fockb[i];
        }
        self.focka = self.fockb.to_vec();
      }
    }

// This constructs the next guess based on the current Fock matrix (which is just the eigenvectors with the lowest eigenvalues)
    pub fn update_evecs(&mut self) {
      let mut info: i32 = 0;

      self.eveca = self.focka.to_vec();
      self.evecb = self.fockb.to_vec();

      unsafe {
        dsyev(b'V',b'U',self.nao as i32,&mut self.eveca,self.nao as i32,
              &mut self.evals,&mut self.work,self.lwork as i32, &mut info);
        dsyev(b'V',b'U',self.nao as i32,&mut self.evecb,self.nao as i32,
                          &mut self.evals,&mut self.work,self.lwork as i32, &mut info);
      }
    }

// Calculate the goodness of the current state which is just measured by taking the commutator of the fock and density matricies.
// do not use this if update_evecs was run after the last call to mk_fock. If you do that this will always return 0.
    pub fn get_err(&mut self) {
      self.err = 0.0;

      unsafe{
        dgemm(b'N',b'N',self.nao as i32,self.nao as i32,self.nao as i32,1.0,
              &self.focka,self.nao as i32,&self.dena,self.nao as i32,0.0,
        &mut self.errvec1,self.nao as i32);
        dgemm(b'N',b'N',self.nao as i32,self.nao as i32,self.nao as i32,1.0,
                          &self.dena,self.nao as i32,&self.focka,self.nao as i32,0.0,
                          &mut self.errvec2,self.nao as i32);
      }

      for i in 0..(self.nao*self.nao) {
        self.err += (self.errvec1[i]-self.errvec2[i]).powi(2);
      }

      unsafe{
                    dgemm(b'N',b'N',self.nao as i32,self.nao as i32,self.nao as i32,1.0,
                          &self.fockb,self.nao as i32,&self.denb,self.nao as i32,0.0,
                          &mut self.errvec1,self.nao as i32);
                    dgemm(b'N',b'N',self.nao as i32,self.nao as i32,self.nao as i32,1.0,
                          &self.denb,self.nao as i32,&self.fockb,self.nao as i32,0.0,
                          &mut self.errvec2,self.nao as i32);
            }

            for i in 0..(self.nao*self.nao) {
                    self.err += (self.errvec1[i]-self.errvec2[i]).powi(2);
            }

    }

// This runs the actual calculation. After construction and initalizing the structure,
// make sure to run mk_guess before calling this so it has some state to work with.
    pub fn do_scf(&mut self,damping: f64, do_uhf: bool) {
      self.err = 1.0;
      let mut niter: usize = 0;


      while self.tol < self.err {
        self.mk_fock(damping,do_uhf);
        self.get_err();
        self.update_evecs();
        self.new_den();
        niter += 1;
        println!("Iteration {}, Energy per particle {}, Error {}",
                 niter, self.get_energy()/((self.nocca+self.noccb) as f64), self.err);
      }
    }


}

// Construct the nearest neighbor hopping matrix for a nx by ny lattice.
// pbcx and pbcy dicate if we use periodic boundary conditions in those respective dimensions.

fn mk_t(nx: usize, ny: usize, pbcx: bool, pbcy: bool) -> Vec<f64> {
  let nao = nx*ny;
  let mut i: usize;
  let mut t = vec![0.0; nao*nao as usize];
  let tmp: bool;

// If the lattice is one dimensional
  if (nx == 1) || (ny == 1) {
    if nx > ny {
      tmp = pbcx;
    } else {
      tmp = pbcy
    }

    i=1;
    while i < nao {
      t[s2i(i-1,i,nao)] = -1.0;
      t[s2i(i,i-1,nao)] = -1.0;
      i += 1;
    }

    if tmp {
      t[s2i(0,nao-1,nao)] = -1.0;
      t[s2i(nao-1,0,nao)] = -1.0;
    }
  } else {
    // if the lattice is 2D
    for i in 0..(nx-1) {
      for j in 0..nx {
        t[s2i(s2i(i,j,nx),s2i(i+1,j,nx),nao)] -= 1.0;
      }
    }
    for i in 1..nx {
      for j in 0..ny {
        t[s2i(s2i(i,j,nx),s2i(i-1,j,nx),nao)] -= 1.0;
      }
    }
    for j in 0..(ny-1) {
      for i in 0..nx {
        t[s2i(s2i(i,j,nx),s2i(i,j+1,nx),nao)] -= 1.0;
      }
    }
    for j in 1..ny {
      for i in 0..nx {
        t[s2i(s2i(i,j,nx),s2i(i,j-1,nx),nao)] -= 1.0;
      }
    }

    if pbcx {
      for j in 0..ny {
        t[s2i(s2i(nx-1,j,nx),s2i(0,j,nx),nao)] -= 1.0;
        t[s2i(s2i(0,j,nx),s2i(nx-1,j,nx),nao)] -= 1.0;
      }
    }

    if pbcy {
      for i in 0..nx {
        t[s2i(s2i(i,ny-1,nx),s2i(i,0,nx),nao)] -= 1.0;
        t[s2i(s2i(i,0,nx),s2i(i,ny-1,nx),nao)] -= 1.0;
      }
    }
  }

  t
}


// This just converts the i-j indicies into a single index so I don't have to remember what goes where.

fn s2i(a: usize, b: usize, n: usize) -> usize {
  a+b*n
}

/*
// does what it says for some debugging
fn display_matrix(a: &Vec<f64>, m: usize, n: usize) {
  let mut i: usize = 0;
  while i < m {
    println!("{:?}", &a[m*i..m*i+n]);
    i += 1;
  }
}
*/


