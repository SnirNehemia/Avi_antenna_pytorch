def normalize_gain(image, gain_pol=None):
    """ 
    input - image of a nin-normalized farfield as a 2d ndarray - first dimension is phi and the second is theta
    output - the normalized farfield
    """
    gain = image
    theta_rad = (torch.linspace(0, 180, gain.shape[1], dtype=torch.float32) * torch.pi / 180)  # get theta vector
    phi_rad = (torch.linspace(0, 360, gain.shape[0], dtype=torch.float32) * torch.pi / 180)  # get phi vector
    d_theta = torch.max(torch.diff(theta_rad))
    d_phi = torch.max(torch.diff(phi_rad))
    efficiency = torch.sum(torch.multiply(gain, torch.sin(theta_rad).unsqueeze(1))) * d_theta * d_phi / (4 * torch.pi)
    directivity = gain / efficiency
    return directivity
    # if gain_pol == None:
    #     return directivity
    # else:
    #     directivity_pol = gain_pol / efficiency
    #     return directivity_pol