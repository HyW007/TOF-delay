import itertools as it
import os
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.units import UnitTypeError, get_physical_type
from astropy.units.quantity import Quantity
from scipy.special import loggamma
from snewpy._model_downloader import LocalFileLoader

from snewpy.neutrino import Flavor
from snewpy.flavor_transformation import NoTransformation
from functools import wraps

from snewpy.flux import Flux
from pathlib import Path

def _wrap_init(init, check):
    @wraps(init)
    def _wrapper(self, *arg, **kwargs):
        init(self, *arg, **kwargs)
        check(self)
    return _wrapper
    
class SupernovaModel(ABC, LocalFileLoader):
    """Base class defining an interface to a supernova model."""

    def __init_subclass__(cls, **kwargs):
        """Hook to modify the subclasses on creation"""
        super().__init_subclass__(**kwargs)
        cls.__init__ = _wrap_init(cls.__init__, cls.__post_init_check)

    def __init__(self, time, metadata):
        """Initialize supernova model base class
        (call this method in the subclass constructor as ``super().__init__(time,metadata)``).

        Parameters
        ----------
        time : ndarray of astropy.Quantity
            Time points where the model flux is defined.
            Must be array of :class:`Quantity`, with units convertable to "second".
        metadata : dict
            Dict of model parameters <name>:<value>,
            to be used for printing table in :meth:`__repr__` and :meth:`_repr_markdown_`
        """
        self.time = time
        self.metadata = metadata
        
    def __repr__(self):
        """Default representation of the model.
        """

        mod = f"{self.__class__.__name__} Model"
        try:
            mod += f': {self.filename}'
        except AttributeError:
            pass
        s = [mod]
        for name, v in self.metadata.items():
            s += [f"{name:16} : {v}"]
        return '\n'.join(s)

    def __post_init_check(self):
        """A function to check model integrity after initialization"""
        try:
            t = self.time
            m = self.metadata
        except AttributeError as e:
            clsname = self.__class__.__name__
            raise TypeError(f"Model not initialized. Please call 'SupernovaModel.__init__' within the '{clsname}.__init__'") from e

    def _repr_markdown_(self):
        """Markdown representation of the model, for Jupyter notebooks.
        """
        mod = f'**{self.__class__.__name__} Model**'
        try:
            mod +=f': {self.filename}'
        except:
            pass
        s = [mod,'']
        if self.metadata:
            s += ['|Parameter|Value|',
                  '|:--------|:----:|']
            for name, v in self.metadata.items():
                try: 
                    s += [f"|{name} | ${v.value:g}$ {v.unit:latex}|"]
                except:
                    s += [f"|{name} | {v} |"]
        return '\n'.join(s)

    def get_time(self):
        """Returns
        -------
        ndarray of astropy.Quantity
            Snapshot times from the simulation
        """
        return self.time

    @abstractmethod
    def get_initial_spectra(self, t, E, flavors=Flavor):
        """Get neutrino spectra at the source.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial spectra.
        flavors: iterable of snewpy.neutrino.Flavor
            Return spectra for these flavors only (default: all)

        Returns
        -------
        initialspectra : dict
            Dictionary of neutrino spectra, keyed by neutrino flavor.
        """
        pass

    def get_initialspectra(self, *args):
        """DO NOT USE! Only for backward compatibility!

        :meta private:
        """
        warn("Please use `get_initial_spectra()` instead of `get_initialspectra()`!", FutureWarning)
        return self.get_initial_spectra(*args)

    def get_tof(self, distance, energy, masses):
        D = distance.to(u.kpc)
        M = masses.to(u.eV)

        # TOF delay formula from arXiv:1006.1889
        tof = 0.57 * (D.value / 10) * (M.value ** 2) * (30 / energy.value) ** 2 * u.ms
        tof=tof.to(u.s)
        #original function follow:
        # tof=D.to(u.m).value/(c.value*(1-(M.value/ energy.to(u.eV).value)**2)**0.5)-D.to(u.m).value/c.value  * u.s
        return tof

    def get_tof_initial_spectra(self,distance, mass, t, E):
        D = distance.to(u.kpc)
        M = mass.to(u.eV)
        if np.isscalar(t.value):
            t = [t.to(u.s).value]*u.s # Convert to list with one element
        if np.isscalar(E.value):
            E = [E.to(u.MeV).value]*u.MeV

        delayed_spectrum = {}
        for flavor in Flavor:
            delayed_spectrum[flavor]=np.zeros((len(t), len(E)))<<1 / (u.erg * u.s)

        for i,e in enumerate(E):
            tof=self.get_tof(D, e, M)
            T=t-tof
            initialspectra=self.get_initial_spectra(T, e)
            delayed_spectrum[Flavor.NU_E][:,i]= initialspectra[Flavor.NU_E]
            delayed_spectrum[Flavor.NU_X][:, i] = initialspectra[Flavor.NU_X]
            delayed_spectrum[Flavor.NU_E_BAR][:, i] = initialspectra[Flavor.NU_E_BAR]
            delayed_spectrum[Flavor.NU_X_BAR][:, i] = initialspectra[Flavor.NU_X_BAR]

        return delayed_spectrum


    def get_transformed_spectra(self, t, E, flavor_xform):

        """Get neutrino spectra after applying oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial and oscillated spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial and oscillated spectra.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        -------
        dict
            Dictionary of transformed spectra, keyed by neutrino flavor.
        """
        D=10*u.kpc
        M=1* u.eV
        #initialspectra = self.get_tof_initial_spectra(D,M, t, E)
        initialspectra = self.get_initial_spectra(t, E)
        transformed_spectra = {}

        #delay = self.get_flying_delay(E, L, m)

        transformed_spectra[Flavor.NU_E] = \
            flavor_xform.prob_ee(t , E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_ex(t , E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_X] = \
            flavor_xform.prob_xe(t , E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_xx(t , E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_E_BAR] = \
            flavor_xform.prob_eebar(t , E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_exbar(t , E) * initialspectra[Flavor.NU_X_BAR]

        transformed_spectra[Flavor.NU_X_BAR] = \
            flavor_xform.prob_xebar(t , E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_xxbar(t , E) * initialspectra[Flavor.NU_X_BAR]

        return transformed_spectra

    def get_tof_transformed_spectra(self, t, E, flavor_xform):

        """Get neutrino spectra after applying oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial and oscillated spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial and oscillated spectra.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        -------
        dict
            Dictionary of transformed spectra, keyed by neutrino flavor.
        """
        D=10*u.kpc
        M=1* u.eV
        initialspectra = self.get_tof_initial_spectra(D,M, t, E)
        #initialspectra = self.get_initial_spectra(t, E)
        transformed_spectra = {}

        #delay = self.get_flying_delay(E, L, m)

        transformed_spectra[Flavor.NU_E] = \
            flavor_xform.prob_ee(t , E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_ex(t , E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_X] = \
            flavor_xform.prob_xe(t , E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_xx(t , E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_E_BAR] = \
            flavor_xform.prob_eebar(t , E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_exbar(t , E) * initialspectra[Flavor.NU_X_BAR]

        transformed_spectra[Flavor.NU_X_BAR] = \
            flavor_xform.prob_xebar(t , E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_xxbar(t , E) * initialspectra[Flavor.NU_X_BAR]

        return transformed_spectra

    def get_tof_transformed_spectra1(self, t, E, flavor_xform):

        """Get neutrino spectra after applying oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial and oscillated spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial and oscillated spectra.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        -------
        dict
            Dictionary of transformed spectra, keyed by neutrino flavor.
        """
        D=10*u.kpc
        M=0.5* u.eV
        initialspectra = self.get_tof_initial_spectra(D,M, t, E)
        #initialspectra = self.get_initial_spectra(t, E)
        transformed_spectra = {}

        #delay = self.get_flying_delay(E, L, m)

        transformed_spectra[Flavor.NU_E] = \
            flavor_xform.prob_ee(t , E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_ex(t , E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_X] = \
            flavor_xform.prob_xe(t , E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_xx(t , E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_E_BAR] = \
            flavor_xform.prob_eebar(t , E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_exbar(t , E) * initialspectra[Flavor.NU_X_BAR]

        transformed_spectra[Flavor.NU_X_BAR] = \
            flavor_xform.prob_xebar(t , E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_xxbar(t , E) * initialspectra[Flavor.NU_X_BAR]

        return transformed_spectra

    '''def get_delayed_flux(self, distance, arrival_time, energies, masses, flavor_xform):
        """
        Calculate the delayed flux of neutrinos accounting for time-of-flight delays.

        Parameters
        ----------
        arrival_time : astropy.Quantity
            Arrival time of neutrinos in the detector (same for all neutrinos, independent of energy).
        energies : array-like
            List of neutrino energies in MeV.
        masses : float
            Neutrino mass in eV.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        delayed_spectrum[index in the array 'masses'][flavor][index in the array 'energies']== flux
        -------
        dict
            Dictionary of delayed spectra, keyed by neutrino flavor.
        """
        D = distance.to(u.kpc)
        c = 299792458 * u.m / u.s
        tof_light = D.to(u.m) / c
        # print(t)
        # print(D)
        AT = arrival_time.to(u.ms)
        # print(T)
        E = energies.to(u.MeV)
        M = masses.to (u.eV)

        if M.isscalar:
            M = [M]

        reslut = {}
        emission_time={}
        delayed_spectrum={}
        for i, mass in enumerate(M):
            emission_time[i] = {}
            for j, energy in enumerate(E):
                # TOF delay formula from arXiv:1006.1889
                tof = 0.57 * D.value / 10 * (mass.value ** 2) * (30 / energy.value) ** 2 * u.ms
                # Calculate the emission time
                emission_time[i][j] = AT - tof
        ''''''

        # Calculate the transformed spectra at the emission time
        for i, mass in enumerate(M):
            delayed_spectrum[i] = {}
            for j, energy in enumerate(E):
                reslut = self.get_transformed_spectra(emission_time[i][j], E, flavor_xform)
                for flavor in Flavor:
                    delayed_spectrum[i][flavor] = reslut[flavor]

        return delayed_spectrum'''


    """def get_delayed_flux(self, distance, arrival_time, energies, masses, flavor_xform):
            
            Calculate the delayed flux of neutrinos accounting for time-of-flight delays.

            Parameters
            ----------
            arrival_time : astropy.Quantity
              Arrival time of neutrinos in the detector (same for all neutrinos, independent of energy).
            energies : array-like
              List of neutrino energies in MeV.
            masses : float
              Neutrino mass in eV.
            flavor_xform : FlavorTransformation
              An instance from the flavor_transformation module.

            Returns
            delayed_spectrum[flavor][index in the array 'energies']== flux
            -------
            dict
                Dictionary of delayed spectra, keyed by neutrino flavor.
            
            D = distance.to(u.kpc)

            # print(t)
            # print(D)
            AT = arrival_time.to(u.ms)
            # print(T)
            M = masses.to (u.eV)

            reslut = {}
            emission_time={}
            delayed_spectrum ={}


            for flavor in Flavor:
                delayed_spectrum[flavor] = np.zeros(len(energies))/ (u.erg * u.s)

    
            for j, energy in enumerate(energies):
                # TOF delay formula from arXiv:1006.1889
                tof = 0.57 * (D.value / 10)* (M.value ** 2) * (30 / energy.value) ** 2 * u.ms
                #tof=D.to(u.m).value/(c.value*(1-(M.value/ energy.to(u.eV).value)**2)**0.5)-D.to(u.m).value/c.value
                #tof=tof * u.s
                  # Calculate the emission time
                emission_time[j] = AT - tof
                  
                   # Calculate the transformed spectra at the emission time
            for flavor in Flavor:
                for j, energy in enumerate(energies):
                    if emission_time[j]<-50.0 *u.ms :
                        result=0.01/(u.erg*u.s)
                    else:
                        reslut = self.get_transformed_spectra(emission_time[j], energies[j],flavor_xform)
                        delayed_spectrum[flavor][j] = reslut[flavor]

            return delayed_spectrum #, emission_time"""


    def get_flux (self, t, E, distance, flavor_xform=NoTransformation()):
        """Get neutrino flux through 1cm^2 surface at the given distance

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate the neutrino spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the the neutrino spectra.
        distance : astropy.Quantity or float (in kpc)
            Distance from supernova.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        -------
        dict
            Dictionary of neutrino fluxes in [neutrinos/(cm^2*erg*s)], 
            keyed by neutrino flavor.

        """
        distance = distance << u.kpc #assume that provided distance is in kpc, or convert
        factor = 1/(4*np.pi*(distance.to('cm'))**2)
        f = self.get_transformed_spectra(t, E, flavor_xform)

        array = np.stack([f[flv] for flv in sorted(Flavor)])
        return  Flux(data=array*factor, flavor=np.sort(Flavor), time=t, energy=E)

    def get_flux_tof (self, t, E, distance, flavor_xform=NoTransformation()):
        """Get neutrino flux through 1cm^2 surface at the given distance

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate the neutrino spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the the neutrino spectra.
        distance : astropy.Quantity or float (in kpc)
            Distance from supernova.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        -------
        dict
            Dictionary of neutrino fluxes in [neutrinos/(cm^2*erg*s)],
            keyed by neutrino flavor.

        """
        distance = distance << u.kpc #assume that provided distance is in kpc, or convert
        factor = 1/(4*np.pi*(distance.to('cm'))**2)
        f = self.get_tof_transformed_spectra(t, E, flavor_xform)

        array = np.stack([f[flv] for flv in sorted(Flavor)])
        return  Flux(data=array*factor, flavor=np.sort(Flavor), time=t, energy=E)


    def get_oscillatedspectra(self, *args):
        """DO NOT USE! Only for backward compatibility!

        :meta private:
        """
        warn("Please use `get_transformed_spectra()` instead of `get_oscillatedspectra()`!", FutureWarning)
        return self.get_transformed_spectra(*args)

def get_value(x):
    """If quantity x has is an astropy Quantity with units, return just the
    value.

    Parameters
    ----------
    x : Quantity, float, or ndarray
        Input quantity.

    Returns
    -------
    value : float or ndarray
    
    :meta private:
    """
    if type(x) == Quantity:
        return x.value
    return x

class PinchedModel(SupernovaModel):
    """Subclass that contains spectra/luminosity pinches"""
    def __init__(self, simtab, metadata):
        """ Initialize the PinchedModel using the data from the given table.

        Parameters
        ----------
        simtab: astropy.Table 
            Should contain columns TIME, {L,E,ALPHA}_NU_{E,E_BAR,X,X_BAR}
            The values for X_BAR may be missing, then NU_X data will be used
        metadata: dict
            Model parameters dict
        """
        if not 'L_NU_X_BAR' in simtab.colnames:
            # table only contains NU_E, NU_E_BAR, and NU_X, so double up
            # the use of NU_X for NU_X_BAR.
            for val in ['L','E','ALPHA']:
                simtab[f'{val}_NU_X_BAR'] = simtab[f'{val}_NU_X']
        # Get grid of model times.
        time = simtab['TIME'] << u.s
        # Set up dictionary of luminosity, mean energy and shape parameter
        # alpha, keyed by neutrino flavor (NU_E, NU_X, NU_E_BAR, NU_X_BAR).
        self.luminosity = {}
        self.meanE = {}
        self.pinch = {}
        for f in Flavor:
            self.luminosity[f] = simtab[f'L_{f.name}'] << u.erg/u.s
            self.meanE[f] = simtab[f'E_{f.name}'] << u.MeV
            self.pinch[f] = simtab[f'ALPHA_{f.name}']
        super().__init__(time, metadata)


    def get_initial_spectra(self, t, E, flavors=Flavor):
        """Get neutrino spectra/luminosity curves before oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial spectra.
        flavors: iterable of snewpy.neutrino.Flavor
            Return spectra for these flavors only (default: all)

        Returns
        -------
        initialspectra : dict
            Dictionary of model spectra, keyed by neutrino flavor.
        """
        #convert input arguments to 1D arrays
        t = u.Quantity(t, ndmin=1)
        E = u.Quantity(E, ndmin=1)
        #Reshape the Energy array to shape [1,len(E)]
        E = np.expand_dims(E, axis=0)

        initialspectra = {}

        # Estimate L(t), <E_nu(t)> and alpha(t). Express all energies in erg.
        E = E.to_value('erg')

        # Make sure input time uses the same units as the model time grid, or
        # the interpolation will not work correctly.
        t = t.to(self.time.unit)

        for flavor in flavors:
            # Use np.interp rather than scipy.interpolate.interp1d because it
            # can handle dimensional units (astropy.Quantity).
            L  = get_value(np.interp(t, self.time, self.luminosity[flavor].to('erg/s')))
            Ea = get_value(np.interp(t, self.time, self.meanE[flavor].to('erg')))
            a  = np.interp(t, self.time, self.pinch[flavor])

            #Reshape the time-related arrays to shape [len(t),1]
            L  = np.expand_dims(L, axis=1)
            Ea = np.expand_dims(Ea,axis=1)
            a  = np.expand_dims(a, axis=1)
            # For numerical stability, evaluate log PDF and then exponentiate.
            result = \
              np.exp(np.log(L) - (2+a)*np.log(Ea) + (1+a)*np.log(1+a)
                    - loggamma(1+a) + a*np.log(E) - (1+a)*(E/Ea)) / (u.erg * u.s)
            #remove bad values
            result[np.isnan(result)] = 0
            result[:, E[0]==0] = 0
            #remove unnecessary dimensions, if E or t was scalar:
            result = np.squeeze(result)
            initialspectra[flavor] = result
        return initialspectra
