const toValidDate = (d) => {
  const date = new Date(d)
  return Number.isNaN(date.getTime()) ? new Date() : date
}

export const formatDate = (d) => toValidDate(d).toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' })
export const formatTime = (d) => toValidDate(d).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
export const formatDateTime = (d) => `${formatDate(d)} · ${formatTime(d)}`
export const timeAgo = (d) => {
  const diff = Date.now() - toValidDate(d).getTime()
  const minute = 60 * 1000
  const hour = 60 * minute
  const day = 24 * hour
  if (diff < minute) return 'just now'
  if (diff < hour) return `${Math.floor(diff / minute)}m ago`
  if (diff < day) return `${Math.floor(diff / hour)}h ago`
  return `${Math.floor(diff / day)}d ago`
}

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
