# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

from GPyOpt.util.general import get_quantiles
import numpy as np

from ...core.interfaces import IModel, IDifferentiable
from ...core.acquisition import Acquisition


class ScaledExpectedImprovement(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], jitter: np.float64 = np.float64(0))-> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        pdf, cdf, u = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        improvement = standard_deviation * (u * cdf + pdf)

        V_Ix=(standard_deviation**2)*(((u*standard_deviation)**2+1.0)*cdf+u*standard_deviation*pdf)-improvement**2
        # u = u*standard_deviation
        #Phi=cdf
        #phi=pdf


        return improvement/np.sqrt(V_Ix)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2.0 * standard_deviation)

        pdf, cdf, u = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        improvement = standard_deviation * (u * cdf + pdf)
        dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx

        V_Ix = (standard_deviation ** 2.0)*(((u * standard_deviation) ** 2 + 1.0) * cdf + u * standard_deviation * pdf) - improvement ** 2

        dscaled_ei_dx=(dimprovement_dx*np.sqrt(V_Ix)-\
                      0.5*improvement*np.power(V_Ix,-0.5)*
                       (2.0*standard_deviation*(((u * standard_deviation) ** 2 + 1.0) *
                                              cdf + u * standard_deviation * pdf)*dstandard_deviation_dx
                        +2.0*improvement*dimprovement_dx))/V_Ix

        return improvement/np.sqrt(V_Ix), dscaled_ei_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)


    def evaluate_with_gradients_II(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.
            
        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2.0 * standard_deviation)

        pdf, cdf, u = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        improvement = standard_deviation * (u * cdf + pdf)
        dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx

        V_Ix = (standard_deviation ** 2.0) * (
            ((u * standard_deviation) ** 2 + 1.0) * cdf + u * standard_deviation * pdf) - improvement ** 2

        dscaled_ei_dx = (dimprovement_dx * np.sqrt(V_Ix) - \
                         0.5 * improvement * np.power(V_Ix, -0.5) *
                         (2.0 * standard_deviation * (((u * standard_deviation) ** 2 + 1.0) *
                                                      cdf + u * standard_deviation * pdf) * dstandard_deviation_dx
                          + 2.0 * improvement * dimprovement_dx)) / V_Ix

        return  improvement / np.sqrt(V_Ix), dscaled_ei_dx,V_Ix,improvement


