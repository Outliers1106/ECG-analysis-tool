from wfdb import processing
import matplotlib.pyplot as plt
import numpy as np
class new_Comparitor(processing.Comparitor):
	def plot(self, sig_style='',pic_size=1,pic_index=1,title=None, figsize=None,return_fig=False,
			 fig=None):
		if fig == None:
			fig=plt.figure(figsize=figsize)
		ax=fig.add_subplot(pic_size,1,pic_index)
		legend = ['Signal',
				  'Matched Reference Annotations (%d/%d)' % (self.tp, self.n_ref),
				  'Unmatched Reference Annotations (%d/%d)' % (self.fn, self.n_ref),
				  'Matched Test Annotations (%d/%d)' % (self.tp, self.n_test),
				  'Unmatched Test Annotations (%d/%d)' % (self.fp, self.n_test)
				  ]

		# Plot the signal if any
		if self.signal is not None:
			ax.plot(self.signal, sig_style)

			# Plot reference annotations
			ax.plot(self.matched_ref_sample,
					self.signal[self.matched_ref_sample], 'ko')
			ax.plot(self.unmatched_ref_sample,
					self.signal[self.unmatched_ref_sample], 'ko',
					fillstyle='none')
			# Plot test annotations
			ax.plot(self.matched_test_sample,
					self.signal[self.matched_test_sample], 'g+')
			ax.plot(self.unmatched_test_sample,
					self.signal[self.unmatched_test_sample], 'rx')

			ax.legend(legend,fontsize=6,loc='lower right')

		# Just plot annotations
		else:
			# Plot reference annotations
			ax.plot(self.matched_ref_sample, np.ones(self.tp), 'ko')
			ax.plot(self.unmatched_ref_sample, np.ones(self.fn), 'ko',
					fillstyle='none')
			# Plot test annotations
			ax.plot(self.matched_test_sample, 0.5 * np.ones(self.tp), 'g+')
			ax.plot(self.unmatched_test_sample, 0.5 * np.ones(self.fp), 'rx')
			ax.legend(legend[1:],fontsize=6,loc='lower right')

		if title:
			ax.set_title(title,fontsize=8)

		#ax.set_xlabel('time/sample')

		fig.show()

		if return_fig:
			return fig, ax, legend



def compare_annotations(ref_sample, test_sample, window_width, signal=None):
    """
    Compare a set of reference annotation locations against a set of
    test annotation locations.
    See the Comparitor class  docstring for more information.
    Parameters
    ----------
    ref_sample : 1d numpy array
        Array of reference sample locations
    test_sample : 1d numpy array
        Array of test sample locations to compare
    window_width : int
        The maximum absolute difference in sample numbers that is
        permitted for matching annotations.
    signal : 1d numpy array, optional
        The original signal of the two annotations. Only used for
        plotting.
    Returns
    -------
    comparitor : Comparitor object
        Object containing parameters about the two sets of annotations
    Examples
    --------
    >>> import wfdb
    >>> from wfdb import processing
    >>> sig, fields = wfdb.rdsamp('sample-data/100', channels=[0])
    >>> ann_ref = wfdb.rdann('sample-data/100','atr')
    >>> xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
    >>> xqrs.detect()
    >>> comparitor = processing.compare_annotations(ann_ref.sample[1:],
                                                    xqrs.qrs_inds,
                                                    int(0.1 * fields['fs']),
                                                    sig[:,0])
    >>> comparitor.print_summary()
    >>> comparitor.plot()
    """
    comparitor = new_Comparitor(ref_sample=ref_sample, test_sample=test_sample,
                            window_width=window_width, signal=signal)
    comparitor.compare()

    return comparitor