import { format, formatDistanceToNow } from 'date-fns'

export const formatDate  = (d) => format(new Date(d), 'MMM dd, yyyy')
export const formatTime  = (d) => format(new Date(d), 'HH:mm')
export const formatDateTime = (d) => format(new Date(d), 'MMM dd, HH:mm')
export const timeAgo     = (d) => formatDistanceToNow(new Date(d), { addSuffix: true })

export const getStatusColor = (status) => {
  const map = {
    Safe:         'green',
    Healthy:      'green',
    online:       'green',
    Warning:      'amber',
    'High Stress':'amber',
    Danger:       'red',
    Critical:     'red',
    Infected:     'red',
    offline:      'gray',
    Unknown:      'gray',
  }
  return map[status] || 'gray'
}

export const getStatusBadgeClass = (status) => {
  const c = getStatusColor(status)
  return {
    green: 'badge-green',
    amber: 'badge-amber',
    red:   'badge-red',
    blue:  'badge-blue',
    gray:  'badge-gray',
  }[c]
}

export const getSensorStatus = (temp, humidity, soil) => {
  if (temp > 45 || soil < 10) return 'Critical'
  if (temp > 38 || soil < 25 || humidity < 30) return 'Warning'
  return 'Safe'
}

export const downloadBlob = (blob, filename) => {
  const url = URL.createObjectURL(blob)
  const a   = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

export const clsx = (...classes) => classes.filter(Boolean).join(' ')
